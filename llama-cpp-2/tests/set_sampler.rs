//! Integration test for [`LlamaContext::set_sampler`] (backend per-seq draft
//! sampler).
//!
//! Needs a real GGUF model and is therefore `#[ignore]`d by default. Set
//! `LLAMA_TEST_MODEL` to the path of any small GGUF (e.g.
//! `qwen2-1_5b-instruct-q4_0.gguf`) and run:
//!
//! ```console
//! LLAMA_TEST_MODEL=/path/to/model.gguf cargo test -p llama-cpp-2 --test set_sampler -- --ignored
//! ```
//!
//! The backend/model are kept on the stack (not in a `static`) so they drop
//! before process exit, avoiding the Metal backend's exit-time teardown
//! assertion.

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

#[test]
#[ignore = "requires LLAMA_TEST_MODEL to point at a GGUF model"]
fn set_sampler_install_and_decode() {
    let model_path = std::env::var("LLAMA_TEST_MODEL")
        .expect("LLAMA_TEST_MODEL must be set to a GGUF model path for this test");
    let backend = LlamaBackend::init().expect("failed to init backend");
    let model = LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default())
        .expect("failed to load model from LLAMA_TEST_MODEL");
    let mut ctx = model
        .new_context(&backend, LlamaContextParams::default())
        .expect("failed to create context");

    // The backend only accepts a sampler chain (`llama_sampler_chain_n > 0`),
    // mirroring llama.cpp's own `tests/test-backend-sampler.cpp`. A bare
    // `top_k` is not a chain and would be rejected. The sampler must outlive its
    // installation in the context, so it is bound here and kept alive for the
    // duration of the test (its last use is the `decode` below).
    let sampler = LlamaSampler::chain_simple([LlamaSampler::top_k(40), LlamaSampler::dist(1234)]);
    // SAFETY: `sampler` is bound for the remainder of the test scope and is not
    // mutated or freed before the `decode` that uses it.
    assert!(
        unsafe { ctx.set_sampler(0, &sampler) },
        "installing a backend sampler chain for seq 0 should be accepted"
    );

    // A decode after installing the sampler must not crash.
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
    ctx.decode(&mut batch)
        .expect("decode after set_sampler should succeed");
}
