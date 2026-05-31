//! Integration test for per-sequence state save/restore wrappers.
//!
//! Needs a real GGUF model and is therefore `#[ignore]`d by default. Set
//! `LLAMA_TEST_MODEL` to the path of any small GGUF (e.g.
//! `qwen2-1_5b-instruct-q4_0.gguf`) and run:
//!
//! ```console
//! LLAMA_TEST_MODEL=/path/to/model.gguf cargo test -p llama-cpp-2 --test state_seq -- --ignored
//! ```
//!
//! Both acceptance criteria — the size/round-trip happy path and the
//! garbage-bytes rejection — live in a single test. `LlamaBackend::init` is a
//! process-wide singleton that can only be called once, and the backend/model
//! are kept on the stack (not in a `static`) so they drop before process exit,
//! avoiding the Metal backend's exit-time teardown assertion.

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::session::SetSeqStateError;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

/// Decode a short prompt into sequence 0 so the context holds real state.
fn decode_short_prompt(model: &LlamaModel, ctx: &mut LlamaContext) {
    let tokens = model
        .str_to_token("The quick brown fox", AddBos::Always)
        .expect("failed to tokenize prompt");
    let mut batch = LlamaBatch::new(512, 1);
    let last = i32::try_from(tokens.len()).expect("prompt too long") - 1;
    for (i, token) in (0_i32..).zip(tokens) {
        batch
            .add(token, i, &[0], i == last)
            .expect("failed to add token");
    }
    ctx.decode(&mut batch).expect("llama_decode() failed");
}

#[test]
#[ignore = "requires LLAMA_TEST_MODEL to point at a GGUF model"]
fn state_seq_save_restore() {
    let model_path = std::env::var("LLAMA_TEST_MODEL")
        .expect("LLAMA_TEST_MODEL must be set to a GGUF model path for this test");
    let backend = LlamaBackend::init().expect("failed to init backend");
    let model = LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default())
        .expect("failed to load model from LLAMA_TEST_MODEL");
    let mut ctx = model
        .new_context(&backend, LlamaContextParams::default())
        .expect("failed to create context");

    decode_short_prompt(&model, &mut ctx);

    // Snapshot sequence 0: size is non-zero and the Vec length matches it.
    let size = ctx.state_seq_get_size(0);
    assert!(size > 0, "non-empty sequence should report a non-zero size");
    let snapshot = ctx.state_seq_get_data(0);
    assert!(
        !snapshot.is_empty(),
        "snapshot of a populated sequence should be non-empty"
    );
    assert_eq!(
        snapshot.len(),
        size,
        "snapshot length must match the reported state_seq_get_size"
    );

    // Clear sequence 0, then restore the snapshot back into it.
    ctx.clear_kv_cache_seq(Some(0), None, None)
        .expect("failed to clear sequence 0");
    let read = ctx
        .state_seq_set_data(&snapshot, 0)
        .expect("restoring a valid snapshot should succeed");
    assert!(read > 0, "restoring a valid snapshot should read > 0 bytes");

    // Truncated / garbage bytes must be rejected (llama.cpp returns 0).
    let garbage = vec![0_u8; 8];
    assert_eq!(
        ctx.state_seq_set_data(&garbage, 0),
        Err(SetSeqStateError::FailedToSet),
        "garbage bytes should fail to restore"
    );
}
