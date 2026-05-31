# MTP (Multi-Token Prediction) — Rust bindings spec for `llama-cpp-2`

Status: proposal / WIP. Author: consumer = `swissarmyhammer` `llama-agent`.

## Goal

Expose enough of llama.cpp's MTP / `draft-mtp` speculative-decoding surface through
the safe `llama-cpp-2` wrapper that an **embedded** consumer (we run our own
decode loop via `llama_decode`, we do **not** run `llama-server`) can implement
in-process MTP self-speculative decoding for Qwen3.6-style models
(`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`). Target: the ~1.5–2× decode speedup the
unsloth card advertises for `--spec-type draft-mtp`.

The orchestration loop (draft → verify → accept) will live in the consumer
(`swissarmyhammer` `llama-agent`), mirroring llama.cpp's
`common_speculative_impl_draft_mtp` (`common/speculative.cpp`). That file is C++
and is **not** part of the C API, so it cannot be called from Rust — we
reimplement it on top of the core API. This spec only covers the **bindings**
`llama-cpp-2`/`llama-cpp-sys-2` must add so the consumer can do that.

## What llama.cpp already provides (this fork, submodule `da3f990a4`, 2026-05-29)

C API in `llama-cpp-sys-2/llama.cpp/include/llama.h`:
- `enum llama_context_type { LLAMA_CONTEXT_TYPE_DEFAULT = 0, LLAMA_CONTEXT_TYPE_MTP = 1 };` (line ~201)
- `llama_context_params.ctx_type` field (line ~345) — selects an MTP context.
- `llama_set_sampler(struct llama_context * ctx, llama_seq_id seq_id, struct llama_sampler * smpl)` (line ~1274) — optional backend (on-GPU) draft sampling.
- `llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max)` (line ~923) — embd>0 allocates the embd buffer; MTP hook batches carry **both** `token` and `embd` (the impl mallocs `batch.token` itself after an `embd`-init).
- state save/restore: `llama_state_seq_get_size/get_data/set_data` (lines ~833/838/848) — for rollback of the draft context on reject.
- `llama_memory_seq_rm` (line ~712), `llama_memory_seq_pos_max`, `llama_get_memory`.
- `llama_get_embeddings_ith` / `_seq` (lines ~1016/1022).

Fork extension header `llama-cpp-sys-2/llama.cpp/src/llama-ext.h` (NOT upstream `llama.h`):
- `void llama_set_embeddings_pre_norm(struct llama_context * ctx, bool value, bool masked);` (line 113)
- `float * llama_get_embeddings_pre_norm(struct llama_context * ctx);` (line 117)
- `float * llama_get_embeddings_pre_norm_ith(struct llama_context * ctx, int32_t i);` (line 120)

The `draft-mtp` impl (`common/speculative.cpp`, struct `common_speculative_impl_draft_mtp`)
uses exactly this set of C calls (the binding surface we must cover): `llama_batch_init`,
`llama_batch_free`, `llama_decode`, `llama_get_embeddings_pre_norm`,
`llama_get_embeddings_pre_norm_ith`, `llama_set_embeddings_pre_norm`,
`llama_set_sampler`, `llama_get_memory`, `llama_memory_seq_pos_max`,
`llama_model_n_embd`, `llama_n_batch`, `llama_get_model`,
`llama_sampler_chain_init/add/default_params`, `llama_sampler_init_top_k`,
`llama_sampler_free`.

## The gaps (what's missing in the Rust layer today)

1. **`llama-ext.h` is not bound.** `llama-cpp-sys-2/wrapper.h` contains only
   `#include "llama.cpp/include/llama.h"`. bindgen's `llama_.*` allowlist would
   pick up the pre-norm functions, but the header is never included, so
   `llama_set_embeddings_pre_norm` / `llama_get_embeddings_pre_norm[_ith]` are
   absent from the generated FFI. (The symbols ARE compiled into libllama — they
   live in `src/`, part of the library — so only the header/bindgen side is
   missing.)
2. **No `ctx_type` setter.** `llama_context_params.ctx_type` exists in the raw
   FFI struct (it's in `llama.h`), but `LlamaContextParams`
   (`llama-cpp-2/src/context/params.rs`) exposes no `with_ctx_type(...)`.
3. **No embd-carrying batch.** The safe `LlamaBatch` wrapper builds
   token-only batches; MTP hook batches need both `token` and `embd` rows set per
   position. Need a way to construct/fill an embd batch (or a dedicated MTP-batch
   type).
4. **No pre-norm hidden-state read.** Even once bound, expose a safe accessor
   returning the pre-norm embedding row(s) as `&[f32]` / `Vec<f32>` (length
   `n_embd`).
5. **No `llama_set_sampler` (backend per-seq sampler).** Optional — only needed
   for backend (GPU) draft sampling; the consumer can start with CPU sampling and
   skip this. List as phase 2.
6. **Seq-level state save/restore** (`llama_state_seq_get_size/get_data/set_data`)
   may not be exposed by the high-level wrapper (`context/session.rs` currently
   wraps whole-context `get_state_size`/`copy_state_data`/`set_state_data`).
   Needed for the draft context's accept/reject rollback. Verify and add seq-level
   variants if absent.

## Required `llama-cpp-2` additions

### 1. Bind `llama-ext.h` (sys crate)
- In `llama-cpp-sys-2/wrapper.h` add `#include "llama.cpp/src/llama-ext.h"`.
- `llama-ext.h` is a **C++** staging header (it pulls in `<map>` and a
  `std::map`-based `llama_memory_breakdown`). bindgen will choke on those. Two
  options:
  - (preferred) add a tiny C-only shim header, e.g. `wrapper_ext.h`, that
    forward-declares **only** the three `pre_norm` `LLAMA_API` functions (plain
    `bool`/`int32_t`/pointer signatures, C-compatible), and `#include` that from
    `wrapper.h`; or
  - include `llama-ext.h` and add bindgen `blocklist_type`/`blocklist_item` for
    the C++ bits (`llama_memory_breakdown*`, `quantize_state_impl`, `std::*`).
- Add `cargo:rerun-if-changed` for whichever header is added.

### 2. `LlamaContextParams::with_ctx_type`
File: `llama-cpp-2/src/context/params.rs`.
```rust
/// Context type. `Mtp` creates a context that runs the model's NextN/MTP head
/// (for draft-mtp speculative decoding); requires a model with MTP layers.
pub enum LlamaContextType { Default, Mtp }

pub fn with_ctx_type(mut self, ctx_type: LlamaContextType) -> Self {
    self.context_params.ctx_type = match ctx_type {
        LlamaContextType::Default => llama_cpp_sys_2::LLAMA_CONTEXT_TYPE_DEFAULT,
        LlamaContextType::Mtp     => llama_cpp_sys_2::LLAMA_CONTEXT_TYPE_MTP,
    };
    self
}
```
(Mirror the existing `get_set!`/`with_*` style in that file.)

### 3. Pre-norm embeddings on `LlamaContext`
File: `llama-cpp-2/src/context.rs` (or a new `context/mtp.rs`).
```rust
/// Output pre-norm hidden states (the row the MTP head consumes). `masked` ==
/// only for tokens with logits != 0. Wraps `llama_set_embeddings_pre_norm`.
pub fn set_embeddings_pre_norm(&mut self, enabled: bool, masked: bool);

/// Pre-norm hidden state for output `i` as `&[f32]` of len `n_embd`.
/// Wraps `llama_get_embeddings_pre_norm_ith`; returns None if unavailable.
pub fn get_embeddings_pre_norm_ith(&self, i: i32) -> Option<&[f32]>;
```
Length is `model.n_embd()`. Safety: the pointer is valid until the next decode;
bound the slice lifetime to `&self` and document the invalidation.

### 4. Embd batch construction
Either extend `LlamaBatch` to allow setting an `embd` row per position (token +
embd both present), or add a dedicated `LlamaMtpBatch` that wraps
`llama_batch_init(n_tokens, n_embd, 1)` and lets the caller set, per slot:
`token[i]`, `embd[i*n_embd..]`, `pos[i]`, `seq_id`, `logits[i]`. Match the impl's
pattern (it `llama_batch_init`s with `embd=n_embd` then also mallocs `batch.token`).

### 5. (phase 2) `llama_set_sampler` + seq-level state
- `LlamaContext::set_sampler(seq_id, &LlamaSampler) -> bool` wrapping
  `llama_set_sampler` (backend draft sampling). Optional.
- Seq-level state: `state_seq_get_size/get_data/set_data` on `LlamaContext`
  for draft-context rollback, if the whole-context state APIs are insufficient.

## How the consumer will use these (informative)

Per generation step, mirroring `common_speculative_impl_draft_mtp`:
1. Two contexts on one model: target (`Default`) + draft (`with_ctx_type(Mtp)`).
   Call `set_embeddings_pre_norm(true, masked=false)` on target,
   `(true, masked=true)` on draft.
2. Draft: feed (token, pre-norm hidden row carried from the previous target
   verify) as an embd batch → MTP head proposes up to `n_max` draft tokens.
3. Verify: run the draft tokens through the target in one batch; read pre-norm
   rows; accept the longest matching prefix.
4. Roll back the draft's recurrent/KV state past rejected drafts; carry the last
   hidden row into the next step.
This integrates with our existing KV-cache reuse + streaming chunking on the
target context.

## Acceptance criteria

- `cargo build`/bindgen emits `llama_set_embeddings_pre_norm`,
  `llama_get_embeddings_pre_norm`, `llama_get_embeddings_pre_norm_ith` in
  `llama_cpp_sys_2`.
- `LlamaContextParams::with_ctx_type(Mtp)` compiles and a Qwen3.6 MTP GGUF loads
  an MTP context without the "context type MTP requested but model doesn't
  contain MTP layers" warning.
- A round-trip test: draft context proposes tokens; pre-norm rows read back have
  length `n_embd` and are finite.
- End-to-end (in the consumer): greedy output with draft-mtp == greedy output
  without it (identical tokens), with measured acceptance > 0 (fewer target
  forward passes than tokens generated).

## Open questions

- Does `da3f990a4` expose everything `draft-mtp` needs purely via `llama.h` +
  `llama-ext.h`, or does any required symbol still live only in `common/`
  (C++, unbindable)? Audit the full `common_speculative_impl_draft_mtp` call list
  (enumerated above) against the public headers before starting.
- Backend sampling (`llama_set_sampler`) worth it, or is CPU top-k draft
  sampling enough for the first cut?
- Should `with_ctx_type(Mtp)` be feature-gated, or always available?
