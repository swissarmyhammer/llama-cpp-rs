//! Gated correctness harness for the MTP (Multi-Token Prediction) orchestration
//! reference — the headline invariant for the whole draft→verify→accept loop.
//!
//! Needs a real Qwen3.6 MTP GGUF and is therefore `#[ignore]`d by default,
//! mirroring the gating in `llama-cpp-2/tests/mtp_round_trip.rs`. Set
//! `LLAMA_MTP_MODEL` to the path of such a model (e.g. one from
//! `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) and run:
//!
//! ```console
//! LLAMA_MTP_MODEL=/path/to/qwen3.6-mtp.gguf cargo test -p mtp --test correctness -- --ignored
//! ```
//!
//! Without the env var the tests are skipped, so the default `cargo test -p mtp`
//! stays green.
//!
//! Scope: this asserts the consumer-side orchestration contract that the binding
//! round-trip test cannot — that the accelerated loop is observationally
//! equivalent to plain greedy generation (`mtp on == mtp off`, token-for-token)
//! while still committing more than one token per target forward pass (the
//! acceptance signal that catches an off-by-one in the `pending_h` pairing).

use anyhow::Result;

use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;

use mtp::{baseline_generate, mtp_generate, MtpParams};

/// The prompt both generators run. Short and deterministic so the test is fast
/// on a real model.
const PROMPT: &str = "The quick brown fox";

/// Tokens to generate. Modest so the test stays fast on a real model while still
/// giving the draft enough room to land several accepted runs.
const N_PREDICT: usize = 40;

/// Load the model named by `LLAMA_MTP_MODEL`. Panics if the env var is unset —
/// the `#[ignore]` attribute already guards the default `cargo test` run, so a
/// caller only reaches this when explicitly running the gated tests.
fn load_model(backend: &LlamaBackend) -> LlamaModel {
    let model_path = std::env::var("LLAMA_MTP_MODEL")
        .expect("LLAMA_MTP_MODEL must be set to a Qwen3.6 MTP GGUF path for this test");
    LlamaModel::load_from_file(backend, &model_path, &LlamaModelParams::default())
        .expect("failed to load model from LLAMA_MTP_MODEL")
}

#[test]
#[ignore = "requires LLAMA_MTP_MODEL to point at a Qwen3.6 MTP GGUF"]
fn mtp_matches_greedy_and_accepts() -> Result<()> {
    let backend = LlamaBackend::init().expect("failed to init backend");
    let model = load_model(&backend);

    let mtp_out = mtp_generate(&backend, &model, PROMPT, N_PREDICT, MtpParams::default())?;
    let baseline = baseline_generate(&backend, &model, PROMPT, N_PREDICT)?;

    // Determinism (the headline invariant): the draft only proposes, the target
    // verifies, so greedy-with-MTP must equal greedy-without token-for-token.
    assert_eq!(
        mtp_out.tokens, baseline,
        "MTP greedy output diverged from plain greedy generation"
    );

    // Acceptance > 0 (h-pairing correctness signal): the draft must have landed
    // at least one accepted token, and the target must have run fewer forward
    // passes than the number of tokens emitted. A collapse to ~0 acceptance here
    // — with determinism still holding — is the signature of an off-by-one in
    // the `pending_h` pairing.
    assert!(
        mtp_out.accepted_total > 0,
        "MTP accepted no draft tokens (accepted_total == 0): \
         the draft is proposing but nothing verifies — suspect h-pairing"
    );
    assert!(
        mtp_out.target_passes < mtp_out.tokens.len(),
        "MTP ran {} target passes for {} tokens; drafting committed no extra \
         tokens per pass (expected target_passes < tokens.len())",
        mtp_out.target_passes,
        mtp_out.tokens.len()
    );

    Ok(())
}

#[test]
#[ignore = "requires LLAMA_MTP_MODEL to point at a Qwen3.6 MTP GGUF"]
fn mtp_generate_smoke() -> Result<()> {
    let backend = LlamaBackend::init().expect("failed to init backend");
    let model = load_model(&backend);

    // A cheap canary distinct from the equivalence assertion: just run the loop
    // for a few tokens and confirm it returns without error and emits exactly
    // `n_predict` tokens (the prompt is short enough that greedy generation does
    // not hit an end-of-generation token first).
    let n_predict = 8;
    let out = mtp_generate(&backend, &model, PROMPT, n_predict, MtpParams::default())?;
    assert_eq!(
        out.tokens.len(),
        n_predict,
        "mtp_generate should emit exactly n_predict tokens for this prompt"
    );

    Ok(())
}
