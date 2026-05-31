//! Carried state and tuning parameters for the MTP orchestration loop.
//!
//! This module defines the data the draft→verify→accept loop carries across
//! generation steps (see `mtp-orchestration.md`) plus the first two phases of
//! the loop — sync/capture and draft — implemented on [`MtpSession`]. These
//! mirror `common_speculative_impl_draft_mtp::process()` and `draft()` in
//! `llama.cpp/common/speculative.cpp`, restricted to a single sequence and CPU
//! top-k sampling.

use anyhow::{Context, Result};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::mtp_batch::LlamaMtpBatch;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use llama_cpp_2::token::LlamaToken;

use crate::helpers::{accept_h_index, draft_should_stop, shift_h_mapping, verify_acceptance};

/// The single sequence id this session drafts on.
///
/// The reference loop is multi-sequence; this reimplementation handles one
/// sequence, so the draft seed and speculative steps always use seq 0. (The
/// mirror uses the caller-supplied `seq_id` to match the target's batch.)
const DRAFT_SEQ_ID: i32 = 0;

/// Tuning parameters for the MTP draft→verify→accept loop.
///
/// Defaults mirror the reference `common_speculative` parameters used for the
/// MTP head: a small per-step draft budget, greedy drafting (`p_min = 0.0`, so
/// the accepted stream matches plain greedy generation), and a top-k 10 draft
/// sampler.
#[derive(Debug, Clone, Copy)]
pub struct MtpParams {
    /// Maximum draft tokens proposed per step.
    pub n_max: usize,
    /// Minimum draft length; shorter drafts are discarded whole.
    pub n_min: usize,
    /// Minimum top-1 probability required to keep drafting.
    pub p_min: f32,
    /// Top-k cutoff for the draft sampler.
    pub top_k: i32,
}

impl Default for MtpParams {
    fn default() -> Self {
        Self {
            n_max: 4,
            n_min: 1,
            p_min: 0.0,
            top_k: 10,
        }
    }
}

/// The result of verifying a draft against the target.
///
/// A verified step always emits at least one token — the target's guaranteed
/// `next_token` (the +1 of speculative decoding) — even when no draft tokens
/// were accepted. The full emitted stream for the step is `accepted` followed
/// by `next_token` (see [`crate::helpers::compose_emitted`]).
#[derive(Debug, Clone)]
pub struct VerifyOutcome {
    /// Number of leading draft tokens the target agreed with (`0..=drafts.len()`).
    pub n_accepted: usize,
    /// The accepted draft prefix (length `n_accepted`).
    pub accepted: Vec<LlamaToken>,
    /// The target's own next token at the position after the last accepted
    /// draft — always present (the speculative-decoding +1).
    pub next_token: LlamaToken,
    /// The position `next_token` occupies: `n_past + n_accepted + 1`, one past
    /// the accepted frontier. The target's KV is decoded through this position,
    /// so the next step resumes at `next_pos`.
    pub next_pos: i32,
}

/// Per-sequence state carried across MTP generation steps.
///
/// `pending_h` is the pre-norm hidden row that pairs with the *next* token fed
/// to the MTP head; `verify_h`/`n_rows` hold the target pre-norm rows captured
/// during the most recent verification decode.
///
/// These fields are written by [`MtpSession::new`] and read/updated by
/// [`MtpSession::sync_capture`] and [`MtpSession::draft`].
#[derive(Debug)]
pub struct MtpSession {
    /// Model embedding dimension; the row width of `pending_h` and `verify_h`.
    pub(crate) n_embd: usize,
    /// Pre-norm hidden row (len `n_embd`) awaiting the next token.
    pub(crate) pending_h: Vec<f32>,
    /// Target pre-norm rows from the last verify decode (`n_rows * n_embd`).
    pub(crate) verify_h: Vec<f32>,
    /// Number of rows currently held in `verify_h`.
    pub(crate) n_rows: usize,
    /// Tuning parameters for the loop.
    pub(crate) params: MtpParams,
}

impl MtpSession {
    /// Create an empty session for a model with embedding dimension `n_embd`.
    ///
    /// `pending_h` starts as a zeroed row of length `n_embd`; `verify_h` is
    /// empty until the first verification decode populates it.
    ///
    /// # Parameters
    /// - `n_embd`: the model's embedding dimension.
    /// - `params`: tuning parameters for the loop.
    #[must_use]
    pub fn new(n_embd: usize, params: MtpParams) -> Self {
        Self {
            n_embd,
            pending_h: vec![0.0; n_embd],
            verify_h: Vec::new(),
            n_rows: 0,
            params,
        }
    }

    /// Capture the target's pre-norm rows and mirror the accepted tokens onto
    /// the draft context (reference `process()`).
    ///
    /// Call this after a target `decode` of `n` sequential positions
    /// (`batch_tokens[k]` at `batch_positions[k]`, all on `seq_id`). It does two
    /// things:
    ///
    /// 1. **Capture** — copies the target's pre-norm row for each of the `n`
    ///    decoded positions into `verify_h`, sets `n_rows = n`, and carries the
    ///    last row forward as `pending_h` (it pairs with the *next* token fed to
    ///    the MTP head).
    /// 2. **Mirror** — replays the same `n` tokens on the draft context with the
    ///    correctness-critical right-shift h-pairing: the MTP head predicts
    ///    token `k+1` from `(token_k, h_k)`, so draft slot 0 is paired with the
    ///    carry `pending_h` from *before* this call, and slot `k >= 1` is paired
    ///    with the target row of position `k - 1` (see [`shift_h_mapping`]).
    ///
    /// The mirror's `pending_h` snapshot is taken *before* step 1 overwrites it,
    /// so slot 0 sees the carry that genuinely precedes `batch_tokens[0]`.
    ///
    /// Mirror positions request no logits — the draft only needs its recurrent
    /// state advanced to match the target, not draft logits, here.
    ///
    /// # Parameters
    /// - `target`: the target context whose pre-norm rows were just produced.
    /// - `draft`: the draft (MTP) context to advance in lockstep.
    /// - `batch_tokens`: the `n` tokens the target decoded, in order.
    /// - `batch_positions`: their positions (same length as `batch_tokens`).
    /// - `seq_id`: the sequence the tokens belong to.
    ///
    /// # Panics
    ///
    /// Panics if `batch_tokens` and `batch_positions` differ in length.
    pub fn sync_capture(
        &mut self,
        target: &LlamaContext,
        draft: &mut LlamaContext,
        batch_tokens: &[LlamaToken],
        batch_positions: &[i32],
        seq_id: i32,
    ) {
        assert_eq!(
            batch_tokens.len(),
            batch_positions.len(),
            "batch_tokens and batch_positions must describe the same positions",
        );
        let n = batch_tokens.len();
        if n == 0 {
            return;
        }

        // Snapshot the carry that pairs with batch_tokens[0] before step 1
        // overwrites pending_h with the freshly captured last row.
        let carry = self.pending_h.clone();

        // Step 1: capture the target's pre-norm rows into verify_h.
        self.n_rows = n;
        self.verify_h.resize(n * self.n_embd, 0.0);
        for i in 0..n {
            let row = target
                .get_embeddings_pre_norm_ith(
                    i32::try_from(i).expect("row index does not fit into i32"),
                )
                .expect("target produced no pre-norm row for a verified position");
            self.verify_h[i * self.n_embd..(i + 1) * self.n_embd].copy_from_slice(row);
        }
        // The last captured row pairs with the next token (cross-call carryover).
        self.pending_h
            .copy_from_slice(&self.verify_h[(n - 1) * self.n_embd..]);

        // Step 2: mirror the accepted tokens onto the draft with the right-shift
        // h-pairing. Slot 0 uses the pre-call carry; slot k>=1 uses verify_h row
        // k-1.
        let mut mirror = LlamaMtpBatch::new(n, self.n_embd);
        for (k, row_index) in shift_h_mapping(n).into_iter().enumerate() {
            let embd = match row_index {
                None => &carry[..],
                Some(r) => &self.verify_h[r * self.n_embd..(r + 1) * self.n_embd],
            };
            mirror
                .add(batch_tokens[k], embd, batch_positions[k], seq_id, false)
                .expect("mirror batch sized for n positions");
        }
        if let Err(err) = draft.decode_mtp(&mut mirror) {
            eprintln!("mtp sync_capture: draft mirror decode failed: {err}");
        }
    }

    /// Produce up to `params.n_max` draft tokens on the draft context
    /// (reference `draft()`).
    ///
    /// Greedy CPU drafting on a single sequence. Each drafted token `k + 1` is
    /// produced from `(token_k, h_k)`: the seed pairs `id_last` with the carried
    /// `pending_h`, and every subsequent step pairs the just-sampled token with
    /// the pre-norm row the draft produced for it. This is the same h-pairing
    /// invariant `sync_capture` mirrors, run forward speculatively.
    ///
    /// Each step keeps the top-1 candidate from a top-k `params.top_k` filter
    /// (matching the reference's top-k 10 draft sampler) and stops via
    /// [`draft_should_stop`] when confidence drops below `params.p_min` or the
    /// per-step budget `params.n_max` is reached. A draft shorter than
    /// `params.n_min` is discarded whole (returns empty).
    ///
    /// # Parameters
    /// - `draft`: the draft (MTP) context to run.
    /// - `id_last`: the last accepted token, the draft's starting point.
    /// - `n_past`: the position `id_last` occupies; drafts extend from here.
    ///
    /// # Returns
    /// The drafted tokens (length in `[0, params.n_max]`), or an empty vector
    /// when the draft is shorter than `params.n_min` or a decode fails.
    ///
    /// # Panics
    ///
    /// Panics if the draft length does not fit into an [`i32`], or if a logits
    /// position yields no candidates or pre-norm row (an internal invariant
    /// broken only by a backend fault).
    #[must_use]
    pub fn draft(
        &mut self,
        draft: &mut LlamaContext,
        id_last: LlamaToken,
        n_past: i32,
    ) -> Vec<LlamaToken> {
        // Seed: pair id_last with the carried pending_h and request logits.
        let mut batch = LlamaMtpBatch::new(1, self.n_embd);
        batch
            .add(id_last, &self.pending_h, n_past, DRAFT_SEQ_ID, true)
            .expect("seed batch has capacity for one position");
        if let Err(err) = draft.decode_mtp(&mut batch) {
            eprintln!("mtp draft: seed decode failed: {err}");
            return Vec::new();
        }

        let mut result: Vec<LlamaToken> = Vec::new();
        // The seed is the only logits position, so it is read at batch index 0;
        // each subsequent single-token batch is likewise read at index 0.
        let i_batch = 0_i32;
        loop {
            let (top1, top1_prob) = self.top1(draft, i_batch);

            // Stop before keeping a token whose confidence fell below p_min.
            if draft_should_stop(
                top1_prob,
                self.params.p_min,
                result.len(),
                self.params.n_max,
            ) {
                break;
            }

            // h_k for the token we are about to keep, read before the next decode
            // invalidates the draft's pre-norm buffer.
            let h_row = draft
                .get_embeddings_pre_norm_ith(i_batch)
                .expect("draft produced no pre-norm row for a logits position")
                .to_vec();

            result.push(top1);
            if result.len() >= self.params.n_max {
                break;
            }

            // Advance the draft one step: pair the kept token with its own h_k.
            let pos = n_past + i32::try_from(result.len()).expect("draft length fits into i32");
            let mut next = LlamaMtpBatch::new(1, self.n_embd);
            next.add(top1, &h_row, pos, DRAFT_SEQ_ID, true)
                .expect("step batch has capacity for one position");
            if let Err(err) = draft.decode_mtp(&mut next) {
                eprintln!("mtp draft: step decode failed: {err}");
                break;
            }
        }

        if result.len() < self.params.n_min {
            return Vec::new();
        }
        result
    }

    /// Verify a draft against the target and produce the step's outcome
    /// (reference verify, caller side in `common/speculative.cpp` /
    /// `examples/speculative-simple/speculative-simple.cpp`).
    ///
    /// Greedy CPU verification on a single sequence. `verify` decodes the batch
    /// `[id_last, draft_0, …, draft_{n-1}]` on the target at positions
    /// `n_past ..= n_past + n` in one [`LlamaBatch`], every position requesting
    /// logits. The target's greedy choice at batch index `i` is the argmax of
    /// its logits — the token the target predicts will *follow* batch entry `i`:
    /// index 0 (`id_last`'s logits) predicts the token that should match
    /// `draft_0`, and index `i` (`draft_{i-1}`'s logits) the one that should
    /// match `draft_i`. The accepted prefix is the longest run where the
    /// target's prediction matches the draft (via [`verify_acceptance`]).
    ///
    /// The target's choice *at* the accepted frontier (batch index `n_accepted`)
    /// is always emitted — the guaranteed +1 of speculative decoding — correct
    /// for partial, zero, and full acceptance, because the batch holds `n + 1`
    /// logit positions. The decode advances the target's KV through `next_pos`
    /// (`n_past + n_accepted + 1`); [`MtpSession::accept`] then rolls the
    /// rejected positions back.
    ///
    /// # Contract
    /// The caller must NOT have decoded `id_last` into the target's KV at
    /// `n_past` — `verify` owns that decode (it is the batch's first entry). The
    /// previous step's decode must leave the target's KV at `n_past` (i.e. the
    /// last accepted token sits *before* `n_past`, the position `id_last` is
    /// about to occupy). On return, the KV is populated through `next_pos`.
    ///
    /// # Parameters
    /// - `target`: the target context to verify on.
    /// - `id_last`: the last accepted token; `verify` decodes it at `n_past` as
    ///   the verify batch's first entry, supplying the logits that predict
    ///   `draft_0`.
    /// - `drafts`: the drafted tokens to verify (must be non-empty).
    /// - `n_past`: the position `id_last` occupies; drafts extend from here.
    /// - `seq_id`: the sequence the drafts belong to.
    ///
    /// # Returns
    /// A [`VerifyOutcome`] with `n_accepted`, the accepted prefix, the
    /// guaranteed `next_token`, and its position `next_pos`.
    ///
    /// # Errors
    /// Returns an error if the batch length does not fit into an [`i32`] or if
    /// the target decode fails.
    ///
    /// # Panics
    ///
    /// Panics if `drafts` is empty, or if a verified position yields no
    /// candidates (an internal invariant broken only by a backend fault).
    pub fn verify(
        &self,
        target: &mut LlamaContext,
        id_last: LlamaToken,
        drafts: &[LlamaToken],
        n_past: i32,
        seq_id: i32,
    ) -> Result<VerifyOutcome> {
        assert!(
            !drafts.is_empty(),
            "verify requires at least one draft token"
        );
        let n = drafts.len();

        // Decode [id_last, draft_0, …, draft_{n-1}] at n_past ..= n_past + n,
        // every position with logits. id_last must be present: its logits are
        // the target's prediction for draft_0, and the batch's n + 1 logit
        // positions make the frontier +1 readable for any n_accepted in 0..=n.
        let mut batch = LlamaBatch::new(n + 1, 1);
        for (i, &token) in std::iter::once(&id_last).chain(drafts).enumerate() {
            let offset = i32::try_from(i).context("verify batch position does not fit into i32")?;
            batch
                .add(token, n_past + offset, &[seq_id], true)
                .context("verify batch sized for id_last plus the draft positions")?;
        }
        target
            .decode(&mut batch)
            .context("target verify decode failed")?;

        // The target's greedy choice at each batch index (argmax logit): one
        // prediction per logit position, so n + 1 entries indexed 0..=n.
        let target_chosen: Vec<LlamaToken> = (0..=n)
            .map(|i| {
                let i_batch = i32::try_from(i).expect("batch index fits into i32");
                target
                    .candidates_ith(i_batch)
                    .max_by(|a, b| a.logit().total_cmp(&b.logit()))
                    .map(|data| data.id())
                    .expect("target produced no candidates for a verified position")
            })
            .collect();

        // Compare each per-position prediction against the draft it should
        // reproduce; the +1 is the target's choice at the frontier.
        let (n_accepted, next_token) = verify_acceptance(&target_chosen, drafts);
        let next_pos = n_past
            + i32::try_from(n_accepted + 1).context("next position does not fit into i32")?;

        Ok(VerifyOutcome {
            n_accepted,
            accepted: drafts[..n_accepted].to_vec(),
            next_token,
            next_pos,
        })
    }

    /// Accept the verified prefix: roll the target KV back to the accepted
    /// frontier and carry the matching pre-norm row forward (reference
    /// `accept()`).
    ///
    /// Two effects:
    ///
    /// 1. **Target KV rollback** — removes the KV range `[accepted_pos, end)`
    ///    from the target's sequence via `clear_kv_cache_seq` (the same call the
    ///    streaming KV-reuse path uses). The clear is inclusive of
    ///    `accepted_pos`, so the caller passes the first *rejected* position —
    ///    the [`VerifyOutcome::next_pos`] from the same step (`n_past +
    ///    n_accepted + 1`). That keeps `id_last`, the `n_accepted` accepted
    ///    drafts, and the guaranteed `next_token` (which `verify` decoded at
    ///    `next_pos - 1`) and drops only the rejected drafts, so the next decode
    ///    resumes at `next_pos`.
    /// 2. **Carry** — sets `pending_h` to the target's pre-norm row for the last
    ///    accepted token, picked by [`accept_h_index`] (mirrors the reference
    ///    `accept()` row choice, clamped to the captured rows).
    ///
    /// No draft-context rollback is performed, and this is deliberate: the next
    /// [`MtpSession::sync_capture`] re-mirrors the canonical accepted sequence
    /// onto the draft, overwriting the draft's redundant auto-regressive
    /// pre-advancement. (`last_n_drafted` is vestigial in the reference; there
    /// is no draft KV to undo here.)
    ///
    /// # Parameters
    /// - `target`: the target context whose KV is rolled back.
    /// - `n_accepted`: number of accepted draft tokens this step.
    /// - `accepted_pos`: the first rejected position — `VerifyOutcome::next_pos`.
    ///   The KV range `[accepted_pos, end)` is removed (inclusive of this
    ///   position), so everything up to and including `next_pos - 1` survives.
    /// - `seq_id`: the sequence to roll back (must be non-negative).
    ///
    /// # Errors
    /// Returns an error if `seq_id` or `accepted_pos` is negative (does not fit
    /// into a `u32`) or if the KV rollback fails.
    pub fn accept(
        &mut self,
        target: &mut LlamaContext,
        n_accepted: usize,
        accepted_pos: i32,
        seq_id: i32,
    ) -> Result<()> {
        let seq = u32::try_from(seq_id).context("seq_id must be non-negative")?;
        let pos = u32::try_from(accepted_pos).context("accepted_pos must be non-negative")?;
        // Drop the rejected draft positions: clear KV [accepted_pos, end)
        // (p0 = accepted_pos inclusive, p1 = None ⇒ to the end). The caller
        // passes next_pos (the first rejected position), so the accepted prefix
        // and next_token survive.
        target
            .clear_kv_cache_seq(Some(seq), Some(pos), None)
            .context("target KV rollback to accepted frontier failed")?;

        // Carry the pre-norm row for the last accepted token forward. In the
        // single-sequence self-driven `mtp_generate` loop this write is
        // immediately superseded by the following `sync_capture`, which re-mirrors
        // and overwrites `pending_h`; this carry is the authoritative h only for
        // drivers that do not re-mirror after accept (matching the reference).
        let row = accept_h_index(n_accepted, self.n_rows);
        self.pending_h
            .copy_from_slice(&self.verify_h[row * self.n_embd..(row + 1) * self.n_embd]);

        Ok(())
    }

    /// The top-1 candidate and its probability at draft batch index `i_batch`.
    ///
    /// Mirrors the reference draft sampler: a top-k `params.top_k` cut followed
    /// by softmax normalization. Returns the argmax token and its post-softmax
    /// probability `p` (used for the `p_min` confidence gate).
    ///
    /// `top_k` sorts the candidates by logit (descending) and truncates to `k`;
    /// `dist` then fills each survivor's softmax `p`. The greedy pick is the
    /// first entry — the highest logit, hence the highest probability — so the
    /// `dist` seed never affects the result (its random draw is unused).
    ///
    /// # Panics
    ///
    /// Panics if the draft produced no candidates at `i_batch`.
    fn top1(&self, draft: &LlamaContext, i_batch: i32) -> (LlamaToken, f32) {
        let mut candidates = LlamaTokenDataArray::from_iter(draft.candidates_ith(i_batch), false);
        candidates.apply_sampler(&LlamaSampler::top_k(self.params.top_k));
        candidates.apply_sampler(&LlamaSampler::dist(0));
        let top = candidates
            .data
            .first()
            .expect("draft produced no candidates");
        (top.id(), top.p())
    }
}
