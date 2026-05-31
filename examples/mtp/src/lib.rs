//! MTP (Multi-Token Prediction) draft→verify→accept orchestration.
//!
//! This crate is the reference reimplementation of the orchestration loop
//! specified in `mtp-orchestration.md` — the consumer-side logic that turns the
//! `LlamaContextType::Mtp` bindings into accelerated generation. It is built as
//! a library so the loop is reusable by both the `mtp` binary and a gated
//! integration test, and so the pure (model-free) decision rules are
//! unit-testable.
//!
//! The crate provides the carried session state ([`MtpSession`]/[`MtpParams`]),
//! the pure helper rules (in [`helpers`]), the single-round-trip binding demo
//! ([`run_round_trip`]), and the full draft→verify→accept generation loop
//! ([`mtp_generate`]) together with the plain-greedy [`baseline_generate`] the
//! equivalence check compares it against.

pub mod helpers;
pub mod session;

use anyhow::{Context, Result};

use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtp_batch::LlamaMtpBatch;
use llama_cpp_2::token::LlamaToken;

use crate::helpers::{compose_emitted, sequential_positions};

pub use session::{MtpParams, MtpSession};

/// The single sequence the generation loop drives.
///
/// Both `mtp_generate` and `baseline_generate` are single-sequence drivers, so
/// every token is decoded on sequence 0 (matching [`MtpSession`]'s own
/// single-sequence assumption).
const SEQ_ID: i32 = 0;

/// The outcome of one MTP draft round-trip.
#[derive(Debug)]
pub struct RoundTrip {
    /// The last prompt token, carried into the draft (MTP) batch.
    pub seed_token: LlamaToken,
    /// The observed length of the target's pre-norm hidden-state row.
    pub pre_norm_len: usize,
    /// The token(s) the draft (MTP) context proposed.
    pub drafted: Vec<LlamaToken>,
}

/// Run one MTP draft round-trip.
///
/// Decodes `prompt` on a fresh target context, reads the target's pre-norm
/// hidden state for the last prompt token, carries `(token, pre_norm_row)` into
/// a [`LlamaMtpBatch`], decodes that on a draft (MTP) context, and greedily
/// reads back the proposed token from the draft's logits.
///
/// # Parameters
/// - `backend`: the initialized llama backend.
/// - `model`: the loaded model; both contexts share it.
/// - `prompt`: the text fed to the target context.
/// - `n_embd`: the model's embedding dimension (the pre-norm row width).
///
/// # Errors
/// Returns an error if either context fails to create, tokenization fails, a
/// decode fails, the target produces no pre-norm row, or the draft produces no
/// candidates.
///
/// # Panics
/// Panics if the prompt is empty (no tokens) or if its token count does not fit
/// into an `i32`.
pub fn run_round_trip(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt: &str,
    n_embd: usize,
) -> Result<RoundTrip> {
    // Target context: standard graph, pre-norm enabled for all tokens.
    let mut target = model
        .new_context(backend, LlamaContextParams::default())
        .with_context(|| "unable to create the target llama_context")?;
    target.set_embeddings_pre_norm(true, false);

    // Draft context: runs the model's MTP/NextN head; pre-norm masked to the
    // tokens whose logits were requested.
    let mut draft = model
        .new_context(
            backend,
            LlamaContextParams::default().with_ctx_type(LlamaContextType::Mtp),
        )
        .with_context(|| "unable to create the draft (MTP) llama_context")?;
    draft.set_embeddings_pre_norm(true, true);

    // Decode the prompt on the target, requesting logits/pre-norm for the
    // final token.
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt:?}"))?;
    let last_index = i32::try_from(tokens.len() - 1).expect("prompt length does not fit into i32");

    let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch.add(token, i, &[0], i == last_index)?;
    }
    target
        .decode(&mut batch)
        .with_context(|| "target decode failed")?;

    // Read the pre-norm hidden state row the MTP head consumes.
    let pre_norm = target
        .get_embeddings_pre_norm_ith(last_index)
        .with_context(|| "target produced no pre-norm row for the last token")?;
    let pre_norm_len = pre_norm.len();
    let pre_norm_row = pre_norm.to_vec();

    let seed_token = *tokens
        .last()
        .expect("prompt must contain at least one token");

    // Carry (token, pre-norm row) into the MTP batch and decode on the draft.
    let mut mtp_batch = LlamaMtpBatch::new(1, n_embd);
    mtp_batch
        .add(seed_token, &pre_norm_row, last_index, 0, true)
        .with_context(|| "failed to fill the MTP batch")?;
    draft
        .decode_mtp(&mut mtp_batch)
        .with_context(|| "draft (MTP) decode failed")?;

    // Greedily read the proposed draft token from the position whose logits we
    // requested.
    let proposed = draft
        .candidates_ith(last_index)
        .max_by(|a, b| a.logit().total_cmp(&b.logit()))
        .map(|data| data.id())
        .with_context(|| "draft produced no candidates")?;

    Ok(RoundTrip {
        seed_token,
        pre_norm_len,
        drafted: vec![proposed],
    })
}

/// The result of an MTP generation run.
///
/// `tokens` is the generated stream (excluding the prompt). `accepted_total`
/// and `target_passes` are the speedup signal: with greedy sampling the stream
/// matches plain greedy generation, but each target forward pass can commit
/// more than one token, so `accepted_total > 0` and `target_passes <
/// tokens.len()` whenever drafting helps (see `mtp-orchestration.md`).
#[derive(Debug)]
pub struct GenerateOutput {
    /// The generated tokens, in order (the prompt is not included).
    pub tokens: Vec<LlamaToken>,
    /// Total draft tokens the target accepted across all steps.
    pub accepted_total: usize,
    /// Number of target forward passes (verify decodes plus greedy fallbacks).
    pub target_passes: usize,
}

/// Create the target and draft (MTP) contexts for a generation session.
///
/// Both contexts share `model`. The target runs the standard graph with
/// unmasked pre-norm (rows for every decoded position); the draft runs the
/// MTP/NextN head with masked pre-norm (rows only where logits were requested).
/// This mirrors the setup in `mtp-orchestration.md`.
///
/// # Errors
/// Returns an error if either context fails to create.
fn setup_contexts<'a>(
    backend: &LlamaBackend,
    model: &'a LlamaModel,
) -> Result<(LlamaContext<'a>, LlamaContext<'a>)> {
    let mut target = model
        .new_context(backend, LlamaContextParams::default())
        .with_context(|| "unable to create the target llama_context")?;
    target.set_embeddings_pre_norm(true, false);

    let mut draft = model
        .new_context(
            backend,
            LlamaContextParams::default().with_ctx_type(LlamaContextType::Mtp),
        )
        .with_context(|| "unable to create the draft (MTP) llama_context")?;
    draft.set_embeddings_pre_norm(true, true);

    Ok((target, draft))
}

/// The target's greedy (argmax-logit) choice at batch index `i_batch`.
///
/// The deterministic sampler the equivalence invariant relies on: the same
/// greedy pick the baseline takes, so the accepted stream matches plain greedy
/// generation.
///
/// # Panics
///
/// Panics if the target produced no candidates at `i_batch` (an internal
/// invariant broken only by a backend fault).
fn target_greedy(target: &LlamaContext, i_batch: i32) -> LlamaToken {
    target.token_data_array_ith(i_batch).sample_token_greedy()
}

/// Decode the prompt on the target and mirror it onto the draft, returning the
/// first sampled target token and the position it will occupy.
///
/// Decodes the whole tokenized prompt on the target requesting logits/pre-norm
/// on **every** position (so the draft's recurrent state is mirrored across the
/// full prompt — the reference `begin()` `pos_max` check), runs `sync_capture`
/// over the prompt positions, and greedily samples the first generated token
/// from the last prompt position's logits.
///
/// # Returns
/// `(id_last, n_past)`: the first sampled token and the position it will occupy
/// (one past the prompt). The target KV is populated through `n_past - 1`, the
/// contract [`MtpSession::verify`] expects.
///
/// # Errors
/// Returns an error if tokenization or the prompt decode fails.
///
/// # Panics
/// Panics if the prompt is empty or its length does not fit into an `i32`.
fn prefill(
    model: &LlamaModel,
    target: &mut LlamaContext,
    draft: &mut LlamaContext,
    session: &mut MtpSession,
    prompt: &str,
) -> Result<(LlamaToken, i32)> {
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt:?}"))?;
    assert!(!tokens.is_empty(), "prompt must contain at least one token");
    let positions = sequential_positions(0, tokens.len());

    // Logits/pre-norm on every position so sync_capture can mirror the whole
    // prompt onto the draft before the first draft() (reference begin()).
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    for (&token, &pos) in tokens.iter().zip(positions.iter()) {
        batch
            .add(token, pos, &[SEQ_ID], true)
            .with_context(|| "prefill batch sized for the prompt")?;
    }
    target
        .decode(&mut batch)
        .with_context(|| "target prompt decode failed")?;

    session.sync_capture(target, draft, &tokens, &positions, SEQ_ID);

    let last_index = i32::try_from(tokens.len() - 1).expect("prompt length fits into i32");
    let id_last = target_greedy(target, last_index);
    let n_past = i32::try_from(tokens.len()).expect("prompt length fits into i32");
    Ok((id_last, n_past))
}

/// Run greedy generation accelerated by the MTP draft→verify→accept loop.
///
/// Sets up a target and draft context on `model` (see [`setup_contexts`]),
/// prefills `prompt`, then runs the loop from `mtp-orchestration.md` until
/// `n_predict` tokens are emitted or the target chooses an end-of-generation
/// token. With greedy sampling the emitted stream equals plain greedy
/// generation ([`baseline_generate`]); the win is that an accepted draft lets
/// one target pass commit several tokens.
///
/// # Loop invariant
/// Each iteration begins with `id_last` not yet decoded into the target KV at
/// `n_past` (the KV is populated through `n_past - 1`) — the contract
/// [`MtpSession::verify`] owns. After a step, `id_last` becomes the target's
/// guaranteed next token and `n_past` advances past the committed frontier.
///
/// In **both** the verify and the empty-draft fallback paths the step ends by
/// `sync_capture`ing the canonical committed tokens (`id_last` followed by any
/// accepted drafts) onto the draft: this captures the target's pre-norm rows
/// into `verify_h`, re-mirrors the draft's recurrent state, and leaves
/// `pending_h` as the row of the last committed token — exactly the `h` the
/// next draft seed pairs with the next token (the h-pairing invariant that an
/// off-by-one would silently collapse).
///
/// # Errors
/// Returns an error if context setup, tokenization, or any decode fails.
///
/// # Panics
/// Panics if the prompt is empty or a position does not fit into an `i32`.
pub fn mtp_generate(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt: &str,
    n_predict: usize,
    params: MtpParams,
) -> Result<GenerateOutput> {
    let (mut target, mut draft) = setup_contexts(backend, model)?;
    let n_embd = usize::try_from(model.n_embd()).expect("n_embd does not fit into a usize");
    let mut session = MtpSession::new(n_embd, params);

    let (mut id_last, mut n_past) = prefill(model, &mut target, &mut draft, &mut session, prompt)?;

    let mut tokens: Vec<LlamaToken> = Vec::with_capacity(n_predict);
    let mut accepted_total = 0;
    let mut target_passes = 0;

    // Emit the first sampled token (the target's greedy choice at the last
    // prompt position — the same token baseline_generate emits first). The loop
    // then drafts *from* id_last to produce the continuation; id_last is the
    // seed, not part of any step's emitted run, so it is pushed exactly once
    // here.
    if n_predict > 0 {
        tokens.push(id_last);
    }
    if n_predict == 0 || model.is_eog_token(id_last) {
        return Ok(GenerateOutput {
            tokens,
            accepted_total,
            target_passes,
        });
    }

    while tokens.len() < n_predict {
        let drafts = session.draft(&mut draft, id_last, n_past);

        // The tokens committed this step after id_last (accepted drafts only),
        // and the target's guaranteed next token. The fallback path produces no
        // accepted drafts but the same shape: a single next token.
        let (accepted, next_token, next_pos) = if drafts.is_empty() {
            let next_token = greedy_step(&mut target, id_last, n_past)?;
            (Vec::new(), next_token, n_past + 1)
        } else {
            let outcome = session.verify(&mut target, id_last, &drafts, n_past, SEQ_ID)?;
            session
                .accept(&mut target, outcome.n_accepted, outcome.next_pos, SEQ_ID)
                .with_context(|| "accepting the verified prefix failed")?;
            accepted_total += outcome.n_accepted;
            (outcome.accepted, outcome.next_token, outcome.next_pos)
        };
        target_passes += 1;

        // Re-mirror the canonical committed sequence [id_last, accepted…] onto
        // the draft (capturing verify_h aligned to positions n_past + k) and
        // leave pending_h as the last committed token's row, ready to pair with
        // next_token. Uses the pre-advance n_past so the rows line up with the
        // verify/prefill decode the target just produced.
        let mirror_tokens: Vec<LlamaToken> = std::iter::once(id_last)
            .chain(accepted.iter().copied())
            .collect();
        let mirror_positions = sequential_positions(n_past, mirror_tokens.len());
        session.sync_capture(
            &target,
            &mut draft,
            &mirror_tokens,
            &mirror_positions,
            SEQ_ID,
        );

        // Emit the accepted prefix plus the guaranteed next token (the +1).
        let emitted = compose_emitted(&accepted, next_token);
        let mut stop = false;
        for token in emitted {
            if tokens.len() >= n_predict {
                break;
            }
            tokens.push(token);
            if model.is_eog_token(token) {
                stop = true;
                break;
            }
        }
        if stop {
            break;
        }

        id_last = next_token;
        n_past = next_pos;
    }

    Ok(GenerateOutput {
        tokens,
        accepted_total,
        target_passes,
    })
}

/// Plain greedy target step: decode `id_last` at `n_past` and return the
/// target's greedy next token.
///
/// The empty-draft fallback. It keeps the same KV bookkeeping as the verify
/// path: `id_last` is decoded into the target KV at `n_past` (advancing it
/// through `n_past`), and the returned token is the target's argmax at that
/// position — the token that will occupy `n_past + 1`. The caller then
/// `sync_capture`s `[id_last]` to capture its pre-norm row and carry it as
/// `pending_h`.
///
/// # Errors
/// Returns an error if the decode fails.
fn greedy_step(target: &mut LlamaContext, id_last: LlamaToken, n_past: i32) -> Result<LlamaToken> {
    let mut batch = LlamaBatch::new(1, 1);
    batch
        .add(id_last, n_past, &[SEQ_ID], true)
        .with_context(|| "greedy fallback batch has capacity for one token")?;
    target
        .decode(&mut batch)
        .with_context(|| "greedy fallback decode failed")?;
    Ok(target_greedy(target, 0))
}

/// Run plain greedy generation on a single target context with MTP disabled.
///
/// The reference stream the equivalence test compares [`mtp_generate`] against:
/// a standard ([`LlamaContextType::Default`]) context, decode the prompt, then
/// greedily sample and feed back one token at a time until `n_predict` tokens
/// are emitted or an end-of-generation token is chosen.
///
/// # Errors
/// Returns an error if context creation, tokenization, or any decode fails.
///
/// # Panics
/// Panics if the prompt is empty or a position does not fit into an `i32`.
pub fn baseline_generate(
    backend: &LlamaBackend,
    model: &LlamaModel,
    prompt: &str,
    n_predict: usize,
) -> Result<Vec<LlamaToken>> {
    let mut target = model
        .new_context(backend, LlamaContextParams::default())
        .with_context(|| "unable to create the baseline llama_context")?;

    let prompt_tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt:?}"))?;
    assert!(
        !prompt_tokens.is_empty(),
        "prompt must contain at least one token"
    );

    let mut batch = LlamaBatch::new(prompt_tokens.len(), 1);
    batch
        .add_sequence(&prompt_tokens, SEQ_ID, false)
        .with_context(|| "baseline prompt batch sized for the prompt")?;
    target
        .decode(&mut batch)
        .with_context(|| "baseline prompt decode failed")?;

    let mut tokens: Vec<LlamaToken> = Vec::with_capacity(n_predict);
    let mut last_index = i32::try_from(prompt_tokens.len() - 1).expect("prompt length fits i32");
    let mut n_past = i32::try_from(prompt_tokens.len()).expect("prompt length fits i32");

    while tokens.len() < n_predict {
        let next = target_greedy(&target, last_index);
        tokens.push(next);
        if model.is_eog_token(next) {
            break;
        }

        let mut step = LlamaBatch::new(1, 1);
        step.add(next, n_past, &[SEQ_ID], true)
            .with_context(|| "baseline step batch has capacity for one token")?;
        target
            .decode(&mut step)
            .with_context(|| "baseline step decode failed")?;
        last_index = 0;
        n_past += 1;
    }

    Ok(tokens)
}
