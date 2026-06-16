//! Pure, model-free helpers for the MTP draft→verify→accept orchestration.
//!
//! These functions encode the decision rules of the loop described in
//! `mtp-orchestration.md` without touching a live context, so they can be
//! exercised by fast unit tests independent of any model.

use llama_cpp_2::token::LlamaToken;

/// Length of the longest prefix where the target and draft tokens agree.
///
/// This is the verify accept rule: walk both slices in lockstep and count
/// matching tokens until the first divergence (or until the shorter slice ends).
///
/// # Parameters
/// - `target`: tokens the target context chose at each verified position.
/// - `draft`: tokens the draft context proposed at the same positions.
///
/// # Returns
/// The number of leading positions where `target[i] == draft[i]`, in
/// `[0, min(target.len(), draft.len())]`.
#[must_use]
pub fn longest_accepted_prefix(target: &[LlamaToken], draft: &[LlamaToken]) -> usize {
    target
        .iter()
        .zip(draft.iter())
        .take_while(|(t, d)| t == d)
        .count()
}

/// The `verify_h` row index to carry forward as `pending_h` after acceptance.
///
/// Mirrors the reference `accept()` row pick: the nextn row for the last
/// accepted token. Clamped to the last available row so it never indexes past
/// the captured rows (and stays valid even when `n_accepted` reaches the row
/// count or no draft tokens were accepted).
///
/// # Parameters
/// - `n_accepted`: number of accepted draft tokens this step.
/// - `n_rows`: number of nextn rows captured in `verify_h`.
///
/// # Returns
/// `min(n_accepted, n_rows.saturating_sub(1))`.
#[must_use]
pub fn accept_h_index(n_accepted: usize, n_rows: usize) -> usize {
    n_accepted.min(n_rows.saturating_sub(1))
}

/// The right-shifted `verify_h` row mapping for the draft mirror batch.
///
/// When the target accepts `n` sequential tokens, the draft context must be
/// fed the same `n` tokens, but each token is paired with the hidden row that
/// *precedes* it (the MTP head predicts token `k+1` from `(token_k, h_k)`). The
/// first mirror slot therefore pairs with the carry `pending_h` left over from
/// before this verify decode (returned as `None`), and slot `k >= 1` pairs with
/// `verify_h` row `k - 1`.
///
/// This mirrors the reference `process()`: `memcpy(batch.embd + 1*n_embd,
/// h_tgt, row_bytes*(n-1))` shifts the captured rows right by one position, and
/// `set_h(i_batch_beg, pending_h)` fills slot 0 with the carry. An off-by-one
/// here silently collapses acceptance to ~0, so it is isolated and unit-tested.
///
/// # Parameters
/// - `n`: number of sequential tokens accepted by the target this step.
///
/// # Returns
/// A vector of length `n`: `[None, Some(0), Some(1), …, Some(n - 2)]`. Slot 0
/// is `None` (use the carry); slot `k` is `Some(k - 1)`. Empty when `n == 0`.
#[must_use]
pub fn shift_h_mapping(n: usize) -> Vec<Option<usize>> {
    (0..n).map(|k| k.checked_sub(1)).collect()
}

/// Resolve a verified step's `n_accepted` and guaranteed next token from the
/// target's per-batch-index greedy choices.
///
/// This encodes the verify index relationship from the reference
/// (`common_sampler_sample_and_accept_n`, `common/sampling.cpp`, and the verify
/// batch layout in `examples/speculative-simple/speculative-simple.cpp`). The
/// target decodes the batch `[id_last, draft_0, …, draft_{n-1}]`, so its greedy
/// choice at batch index `i` predicts the token that should *follow* batch
/// entry `i`:
///
/// - `target_chosen[0]` (from `id_last`'s logits) is the target's prediction
///   for the token after `id_last`, i.e. the one that should match `draft_0`.
/// - `target_chosen[i]` (from `draft_{i-1}`'s logits) should match `draft_i`.
/// - `target_chosen[n]` (from `draft_{n-1}`'s logits) is the target's choice
///   after the whole draft — only consulted on a full match.
///
/// The accepted prefix is the longest run where `target_chosen[i] == draft_i`
/// (`i` in `0..n`); the guaranteed next token (the speculative-decoding +1) is
/// the target's choice *at* the accepted frontier, `target_chosen[n_accepted]`.
/// Because `target_chosen` has `n + 1` entries this is well-defined for every
/// `n_accepted` in `0..=n` — partial, zero, and full acceptance alike.
///
/// # Parameters
/// - `target_chosen`: the target's greedy choice at each batch index, length
///   `drafts.len() + 1` (one prediction per verified logit position).
/// - `drafts`: the drafted tokens proposed for this step (length `n`).
///
/// # Returns
/// `(n_accepted, next_token)`: the accepted-prefix length in `0..=n` and the
/// target's guaranteed next token at the frontier.
///
/// # Panics
///
/// Panics unless `target_chosen.len() == drafts.len() + 1`.
#[must_use]
pub fn verify_acceptance(
    target_chosen: &[LlamaToken],
    drafts: &[LlamaToken],
) -> (usize, LlamaToken) {
    assert_eq!(
        target_chosen.len(),
        drafts.len() + 1,
        "verify expects one target choice per logit position (drafts.len() + 1)",
    );
    // Compare the target's per-position predictions (the first n entries)
    // against the drafts they should reproduce.
    let n_accepted = longest_accepted_prefix(&target_chosen[..drafts.len()], drafts);
    // The +1 is the target's choice at the frontier — correct for partial,
    // zero, and full acceptance because target_chosen has n + 1 entries.
    (n_accepted, target_chosen[n_accepted])
}

/// Compose a verified step's emitted tokens: the accepted draft prefix plus
/// the target's guaranteed next token.
///
/// This is the data-level shape of speculative decoding's "+1" invariant: a
/// verified step always emits the `n_accepted` draft tokens the target agreed
/// with, followed by the target's own next token (the token it chose at the
/// position after the last accepted draft). Even when `n_accepted == 0` the
/// step still emits exactly one token — `next_token` — so the loop always makes
/// forward progress.
///
/// # Parameters
/// - `accepted`: the accepted draft prefix (length `n_accepted`).
/// - `next_token`: the target's guaranteed next token (the +1).
///
/// # Returns
/// A vector of length `accepted.len() + 1`: the accepted prefix followed by
/// `next_token`.
#[must_use]
pub fn compose_emitted(accepted: &[LlamaToken], next_token: LlamaToken) -> Vec<LlamaToken> {
    let mut emitted = accepted.to_vec();
    emitted.push(next_token);
    emitted
}

/// The `count` sequential decode positions starting at `start`.
///
/// The MTP loop repeatedly builds batches whose tokens occupy consecutive
/// positions on a single sequence — the prompt prefill, the canonical
/// accepted-token mirror, and the greedy fallback all need `[start, start + 1,
/// …, start + count - 1]`. This isolates that index arithmetic so the loop body
/// never open-codes a position range (and so the `i32` conversion is checked in
/// one place).
///
/// # Parameters
/// - `start`: the position of the first token.
/// - `count`: how many sequential positions to produce.
///
/// # Returns
/// A vector of length `count`: `[start, start + 1, …, start + count - 1]`.
/// Empty when `count == 0`.
///
/// # Panics
///
/// Panics if any position does not fit into an [`i32`].
#[must_use]
pub fn sequential_positions(start: i32, count: usize) -> Vec<i32> {
    (0..count)
        .map(|k| {
            start
                .checked_add(i32::try_from(k).expect("position offset fits into i32"))
                .expect("decode position fits into i32")
        })
        .collect()
}

/// Whether the draft loop should stop after the most recent draft token.
///
/// The draft stops when the model is no longer confident enough (top-1
/// probability below `p_min`) or the per-step draft budget is exhausted
/// (`drafted_len >= n_max`).
///
/// # Parameters
/// - `top1_prob`: probability of the top-1 draft candidate at this position.
/// - `p_min`: minimum top-1 probability required to keep drafting.
/// - `drafted_len`: number of tokens drafted so far this step.
/// - `n_max`: maximum draft tokens allowed per step.
///
/// # Returns
/// `true` when drafting should stop, `false` to continue.
#[must_use]
pub fn draft_should_stop(top1_prob: f32, p_min: f32, drafted_len: usize, n_max: usize) -> bool {
    top1_prob < p_min || drafted_len >= n_max
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(ids: &[i32]) -> Vec<LlamaToken> {
        ids.iter().copied().map(LlamaToken).collect()
    }

    #[test]
    fn longest_accepted_prefix_empty_inputs() {
        assert_eq!(longest_accepted_prefix(&[], &[]), 0);
        assert_eq!(longest_accepted_prefix(&toks(&[1, 2]), &[]), 0);
        assert_eq!(longest_accepted_prefix(&[], &toks(&[1, 2])), 0);
    }

    #[test]
    fn longest_accepted_prefix_full_match() {
        assert_eq!(
            longest_accepted_prefix(&toks(&[1, 2, 3]), &toks(&[1, 2, 3])),
            3
        );
    }

    #[test]
    fn longest_accepted_prefix_partial_match() {
        assert_eq!(
            longest_accepted_prefix(&toks(&[1, 2, 9]), &toks(&[1, 2, 3])),
            2
        );
    }

    #[test]
    fn longest_accepted_prefix_divergent_at_zero() {
        assert_eq!(
            longest_accepted_prefix(&toks(&[9, 2, 3]), &toks(&[1, 2, 3])),
            0
        );
    }

    #[test]
    fn longest_accepted_prefix_stops_at_shorter_slice() {
        // The shorter slice bounds the prefix even when every shared position agrees.
        assert_eq!(
            longest_accepted_prefix(&toks(&[1, 2]), &toks(&[1, 2, 3])),
            2
        );
    }

    #[test]
    fn accept_h_index_accepted_below_rows() {
        assert_eq!(accept_h_index(1, 4), 1);
    }

    #[test]
    fn accept_h_index_accepted_equals_rows_clamps() {
        assert_eq!(accept_h_index(4, 4), 3);
    }

    #[test]
    fn accept_h_index_no_rows() {
        assert_eq!(accept_h_index(0, 0), 0);
        assert_eq!(accept_h_index(3, 0), 0);
    }

    #[test]
    fn accept_h_index_single_row() {
        assert_eq!(accept_h_index(0, 1), 0);
        assert_eq!(accept_h_index(5, 1), 0);
    }

    #[test]
    fn shift_h_mapping_typical() {
        // Slot 0 uses the carry; slot k pairs with verify_h row k-1.
        assert_eq!(shift_h_mapping(4), vec![None, Some(0), Some(1), Some(2)],);
    }

    #[test]
    fn shift_h_mapping_single_row() {
        // One accepted token pairs only with the carry — no captured row.
        assert_eq!(shift_h_mapping(1), vec![None]);
    }

    #[test]
    fn shift_h_mapping_empty() {
        assert_eq!(shift_h_mapping(0), Vec::<Option<usize>>::new());
    }

    #[test]
    fn shift_h_mapping_indices_are_one_behind_slot() {
        // The off-by-one guard: every slot k>=1 must map to row k-1, and only
        // slot 0 may be the carry.
        let map = shift_h_mapping(8);
        assert_eq!(map[0], None);
        for (slot, entry) in map.iter().enumerate().skip(1) {
            assert_eq!(*entry, Some(slot - 1));
        }
    }

    #[test]
    fn verify_acceptance_full_match_takes_frontier_next() {
        // Drafts [1,2,3]; target predicts each correctly and then chooses 4 at
        // batch index 3 (draft_2's logits). All three accepted, +1 is 4.
        let target_chosen = toks(&[1, 2, 3, 4]);
        let drafts = toks(&[1, 2, 3]);
        assert_eq!(
            verify_acceptance(&target_chosen, &drafts),
            (3, LlamaToken(4))
        );
    }

    #[test]
    fn verify_acceptance_partial_match_next_is_frontier_choice() {
        // Drafts [1,2,3]; target agrees on the first two, diverges at index 2
        // (chooses 9 where the draft said 3). n_accepted = 2 and the +1 is the
        // divergent choice 9 — read at the frontier index, not index 2+1.
        let target_chosen = toks(&[1, 2, 9, 99]);
        let drafts = toks(&[1, 2, 3]);
        assert_eq!(
            verify_acceptance(&target_chosen, &drafts),
            (2, LlamaToken(9))
        );
    }

    #[test]
    fn verify_acceptance_zero_match_next_is_id_last_prediction() {
        // The target's choice after id_last (batch index 0) disagrees with
        // draft_0 immediately. n_accepted = 0 and the +1 is that index-0 choice
        // (5) — the token at n_past+1, not the index-1 prediction.
        let target_chosen = toks(&[5, 6, 7]);
        let drafts = toks(&[1, 2]);
        assert_eq!(
            verify_acceptance(&target_chosen, &drafts),
            (0, LlamaToken(5))
        );
    }

    #[test]
    fn verify_acceptance_single_draft_accepted() {
        // One draft, accepted: +1 comes from batch index 1 (draft_0's logits).
        let target_chosen = toks(&[1, 2]);
        let drafts = toks(&[1]);
        assert_eq!(
            verify_acceptance(&target_chosen, &drafts),
            (1, LlamaToken(2))
        );
    }

    #[test]
    fn verify_acceptance_single_draft_rejected() {
        // One draft, rejected at index 0: +1 is the index-0 choice (9).
        let target_chosen = toks(&[9, 8]);
        let drafts = toks(&[1]);
        assert_eq!(
            verify_acceptance(&target_chosen, &drafts),
            (0, LlamaToken(9))
        );
    }

    #[test]
    #[should_panic(expected = "one target choice per logit position")]
    fn verify_acceptance_rejects_mismatched_lengths() {
        // target_chosen must be drafts.len() + 1.
        let _ = verify_acceptance(&toks(&[1, 2]), &toks(&[1, 2]));
    }

    #[test]
    fn compose_emitted_appends_next_after_full_prefix() {
        // Full acceptance: every accepted draft token, then the guaranteed +1.
        assert_eq!(
            compose_emitted(&toks(&[1, 2, 3]), LlamaToken(4)),
            toks(&[1, 2, 3, 4]),
        );
    }

    #[test]
    fn compose_emitted_partial_prefix() {
        // Partial acceptance still ends with the target's next token.
        assert_eq!(
            compose_emitted(&toks(&[1, 2]), LlamaToken(9)),
            toks(&[1, 2, 9]),
        );
    }

    #[test]
    fn compose_emitted_zero_accepted_still_emits_next() {
        // The +1 invariant: even with no accepted drafts, exactly one token
        // (the target's next token) is emitted, so the loop makes progress.
        assert_eq!(compose_emitted(&[], LlamaToken(7)), toks(&[7]));
    }

    #[test]
    fn sequential_positions_typical() {
        assert_eq!(sequential_positions(5, 3), vec![5, 6, 7]);
    }

    #[test]
    fn sequential_positions_single() {
        assert_eq!(sequential_positions(0, 1), vec![0]);
    }

    #[test]
    fn sequential_positions_empty() {
        assert_eq!(sequential_positions(9, 0), Vec::<i32>::new());
    }

    #[test]
    fn draft_should_stop_below_p_min() {
        assert!(draft_should_stop(0.4, 0.5, 0, 4));
    }

    #[test]
    fn draft_should_stop_at_p_min_continues() {
        // Exactly meeting p_min is not below it, so drafting continues.
        assert!(!draft_should_stop(0.5, 0.5, 0, 4));
    }

    #[test]
    fn draft_should_stop_at_n_max() {
        assert!(draft_should_stop(0.99, 0.0, 4, 4));
    }

    #[test]
    fn draft_should_stop_below_n_max_continues() {
        assert!(!draft_should_stop(0.99, 0.0, 3, 4));
    }

    #[test]
    fn draft_should_stop_past_n_max() {
        assert!(draft_should_stop(0.99, 0.0, 5, 4));
    }
}
