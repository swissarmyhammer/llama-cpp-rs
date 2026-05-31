# MTP draft→verify→accept orchestration — algorithm spec

Companion to [`mtp.md`](./mtp.md). `mtp.md` specified the **bindings**; those are now
implemented (`with_ctx_type`, `context::mtp` pre-norm accessors, `LlamaMtpBatch` +
`decode_mtp`, `set_sampler`, `state_seq_{get_size,get_data,set_data}`), and
`examples/mtp` proves the binding **round-trip** (target → pre-norm row → draft
MTP batch → proposed token).

What is NOT in the bindings — and is the subject of this doc — is the
**orchestration loop**: how to use those bindings to actually accelerate
generation. It mirrors `common_speculative_impl_draft_mtp` in
`llama-cpp-sys-2/llama.cpp/common/speculative.cpp` (C++, not part of the C API,
so it cannot be linked — it is reimplemented by the consumer). This doc is the
canonical reference for that reimplementation and lives next to the bindings so
anyone using `LlamaContextType::Mtp` has the algorithm at hand.

Consumer: `swissarmyhammer` `llama-agent` `GenerationHelper`. This is a design
spec, not consumer code.

## Setup (once per session)

Two contexts on the **same** loaded model:
- **target** — `LlamaContextParams::with_ctx_type(Default)`; `set_embeddings_pre_norm(true, masked=false)`.
- **draft** — `with_ctx_type(Mtp)`; `set_embeddings_pre_norm(true, masked=true)`.

`n_embd = model.n_embd()`. Per-seq state carried across steps:
- `pending_h: Vec<f32>` (len `n_embd`) — the pre-norm hidden row that pairs with
  the NEXT token fed to the MTP head. The MTP head predicts token `p+1` from
  `(x_{p+1}, h_p)`; `pending_h` is `h_p` waiting for `x_{p+1}`.
- `verify_h: Vec<f32>` (`n_rows × n_embd`) and `n_rows` — target pre-norm rows
  from the most recent verification decode. Row 0 = the sampled token's row, row
  k = the k-th accepted draft token's row.

Reference params: `n_max` (max draft tokens/step, e.g. 2–4), `n_min` (drop the
whole draft if shorter), `p_min` (min top-1 prob to keep drafting), draft sampler
= top-k 10.

## Per-turn loop

For each generation step on the target:

### 1. Sync + capture (the reference `process()` hook)
After a target `decode` that advances accepted tokens:
- Read the target's pre-norm rows for the decoded positions via
  `target.get_embeddings_pre_norm_ith(i)` → fill `verify_h` (`n_rows` = number of
  rows), and set `pending_h = verify_h[n_rows-1]` (carryover: last row pairs with
  the next token).
- Mirror those accepted (token, h) pairs onto the **draft** context with a
  `LlamaMtpBatch` + `decode_mtp` so the draft's recurrent state stays aligned
  with the canonical accepted sequence. NOTE (reference behavior): re-mirroring
  the verified batch each round is ALSO how the draft's redundant
  auto-regressive pre-advancement from the previous `draft()` is rolled back —
  `last_n_drafted` exists only to reason about this. In our single-seq driver
  this re-mirroring is sufficient: no explicit draft rollback
  (`state_seq_set_data` or `clear_kv_cache_seq` on the draft) is needed (see
  Resolved #1).

### 2. Draft (the reference `draft()`)
Produce up to `n_max` draft tokens on the **draft** context:
1. Seed: `LlamaMtpBatch::new(n, n_embd)`; add `(id_last, pos=n_past, logits=true)`
   and copy `pending_h` into that position's `embd` row. `draft.decode_mtp(batch)`.
2. Loop: sample the draft logits (top-k 10) at the just-decoded row; read that
   row's pre-norm via `draft.get_embeddings_pre_norm_ith(i_batch)` → `h_row`.
   - If top-1 prob `< p_min` → stop drafting.
   - Else append the token to `result`. If `result.len() >= n_max` → stop.
   - Else add `(drafted_id, pos=n_past+i+1, logits=true)` to a fresh batch and
     copy `h_row` into its `embd`. `decode_mtp`. Repeat.
3. If `result.len() < n_min` → discard the whole draft (empty).

Each drafted token `k+1` is produced from `(token_k, h_k)` where `h_k` is the
pre-norm row produced when the draft decoded `token_k`; the first uses
`pending_h`.

### 3. Verify (target)
Decode the drafted tokens on the **target** in ONE batch (after `id_last`),
requesting logits for each. Greedily (or per the real sampler) read the target's
chosen token at each position; accept the longest prefix where target == draft.
`n_accepted` ∈ `[0, result.len()]`. The token after the last accepted draft (the
target's own next token) is always produced — that is the +1 of speculative
decoding.

### 4. Accept / rollback
- Roll the **target** KV back to the accepted length: `clear_kv_cache_seq(Some(0),
  Some(accepted_pos), None)` to drop rejected draft positions (same call our
  streaming KV-reuse already uses).
- Set `pending_h = verify_h[min(n_accepted, n_rows-1)]` — the target's pre-norm
  row for the last accepted token (the reference `accept()`).
- Continue at step 1 from the new accepted frontier.

## Binding → phase map

| Phase | Bindings used |
|-------|---------------|
| setup | `with_ctx_type(Mtp)`, `set_embeddings_pre_norm` |
| sync/capture | target `decode`, `get_embeddings_pre_norm_ith`, draft `decode_mtp` |
| draft | `LlamaMtpBatch::new`/add, `decode_mtp`, `get_embeddings_pre_norm_ith`, sampler (top-k) / optional `set_sampler` |
| verify | target `decode` (batch of drafts) + logits |
| accept | `clear_kv_cache_seq` (target rollback), `verify_h` row copy (no draft rollback — re-mirroring suffices, see Resolved #1) |

## Correctness invariants & verification

- **Determinism:** with greedy sampling, the accepted token stream MUST equal
  plain greedy generation without MTP — the draft only proposes; the target
  verifies. The headline test: same prompt, greedy, `mtp on == mtp off` token-for-
  token.
- **Speedup signal (non-timing):** measured acceptance > 0, i.e. fewer target
  forward passes than tokens emitted. Log accepted/total per turn.
- **h-pairing:** an off-by-one in `pending_h` (pairing `h_p` with the wrong token)
  silently degrades acceptance to ~0 without breaking correctness — assert
  acceptance > 0 on a real model to catch it.
- Integrate with our existing streaming KV-cache reuse on the **target** context;
  the draft context is ephemeral per session.

## Open questions / verify against the reference — Resolved

Resolved against the reference `common_speculative_impl_draft_mtp` while building
the orchestration loop and its gated correctness harness
(`examples/mtp/tests/correctness.rs`).

1. **Draft rollback — Resolved.** Re-mirroring in `sync_capture` is sufficient;
   no explicit `state_seq_set_data` / `clear_kv_cache_seq` on the draft is needed
   for our single-sequence, self-driven loop. The reference's `last_n_drafted`
   bookkeeping is vestigial — it exists to support the generic `common` driver's
   rollback, which we do not use.
2. **Prefill capture — Resolved (confirmed).** The driver must decode the full
   prompt on the target with pre-norm/logits requested on **every** position and
   `sync_capture` it onto the draft before the first `draft()`. This mirrors the
   reference running `process()` on every target ubatch including the prefill,
   and is what populates `verify_h`/`pending_h` for the first draft seed.
3. **`need_embd()` — Resolved.** `need_embd()` returning false (and
   `need_embd_pre_norm()` true) only toggles what the generic `common` driver
   auto-captures. When we drive manually we call `set_embeddings_pre_norm`
   ourselves, so it does not matter for this reference.
4. **Backend draft sampling — Resolved.** CPU top-k is chosen for this reference;
   `set_sampler` is left available but unused. Backend sampling can be revisited
   if measurement later shows it worthwhile.
