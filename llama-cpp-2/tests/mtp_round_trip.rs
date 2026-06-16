//! Gated integration test for the MTP (Multi-Token Prediction) binding
//! surface: target/draft contexts, nextn accessors, [`LlamaMtpBatch`], and
//! [`LlamaContext::decode_mtp`].
//!
//! Needs a real Qwen3.6 MTP GGUF and is therefore `#[ignore]`d by default. Set
//! `LLAMA_MTP_MODEL` to the path of such a model (e.g. one from
//! `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) and run:
//!
//! ```console
//! LLAMA_MTP_MODEL=/path/to/qwen3.6-mtp.gguf cargo test -p llama-cpp-2 --test mtp_round_trip -- --ignored
//! ```
//!
//! Without the env var the test is skipped, so the default
//! `cargo test -p llama-cpp-2` stays green.
//!
//! Scope: this asserts only the binding round-trip — an MTP context loads
//! without the missing-MTP-layers warning, a nextn row has length `n_embd`
//! with all-finite values, and the draft proposes at least one token. The
//! broader end-to-end equivalence check (greedy-with-mtp == greedy-without,
//! acceptance rate > 0) is a consumer concern and lives in `swissarmyhammer`'s
//! `llama-agent`, because the accept/verify loop is not part of these bindings.

use std::ffi::{c_void, CStr};
use std::os::raw::c_char;
use std::sync::Mutex;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaContextType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::mtp_batch::LlamaMtpBatch;

/// The warning llama.cpp emits when an MTP context is requested against a model
/// that has no MTP/NextN layers. Its absence is part of the acceptance
/// criteria.
const MISSING_MTP_WARNING: &str = "context type MTP requested but model doesn't contain MTP layers";

/// Captures llama.cpp log text so the test can assert which messages were (and
/// were not) emitted while the MTP context was created.
static CAPTURED_LOGS: Mutex<String> = Mutex::new(String::new());

/// `llama_log_set` callback that appends every log message into [`CAPTURED_LOGS`].
unsafe extern "C" fn capture_log(
    _level: llama_cpp_sys_2::ggml_log_level,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: llama.cpp passes a valid NUL-terminated C string for each chunk.
    let chunk = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    if let Ok(mut buf) = CAPTURED_LOGS.lock() {
        buf.push_str(&chunk);
    }
}

#[test]
#[ignore = "requires LLAMA_MTP_MODEL to point at a Qwen3.6 MTP GGUF"]
fn mtp_draft_context_round_trip() {
    let model_path = std::env::var("LLAMA_MTP_MODEL")
        .expect("LLAMA_MTP_MODEL must be set to a Qwen3.6 MTP GGUF path for this test");

    let backend = LlamaBackend::init().expect("failed to init backend");
    let model = LlamaModel::load_from_file(&backend, &model_path, &LlamaModelParams::default())
        .expect("failed to load model from LLAMA_MTP_MODEL");

    let n_embd = usize::try_from(model.n_embd()).expect("n_embd does not fit into a usize");

    // Target context: standard graph, nextn enabled for all tokens.
    let mut target = model
        .new_context(&backend, LlamaContextParams::default())
        .expect("failed to create target context");
    target.set_embeddings_nextn(true, false);

    // Install a capturing log sink so we can assert the missing-MTP-layers
    // warning is absent while the MTP context is created. Cleared first in case
    // an earlier run left content behind.
    CAPTURED_LOGS.lock().unwrap().clear();
    unsafe {
        llama_cpp_sys_2::llama_log_set(Some(capture_log), std::ptr::null_mut());
    }

    // Draft context: runs the model's MTP/NextN head. Creating it must succeed
    // and must NOT emit the missing-MTP-layers warning for a real MTP model.
    let mtp_result = model.new_context(
        &backend,
        LlamaContextParams::default().with_ctx_type(LlamaContextType::Mtp),
    );

    let logged = CAPTURED_LOGS.lock().unwrap().clone();
    // Restore the default (stderr) log sink before any assertion can unwind, so
    // later teardown logs don't dereference our (soon-gone) callback.
    unsafe {
        llama_cpp_sys_2::llama_log_set(None, std::ptr::null_mut());
    }

    let mut draft = mtp_result.expect("creating an MTP context on an MTP model should succeed");
    assert!(
        !logged.contains(MISSING_MTP_WARNING),
        "MTP context creation logged the missing-MTP-layers warning; \
         the model at LLAMA_MTP_MODEL does not contain MTP layers.\nlogs:\n{logged}"
    );

    draft.set_embeddings_nextn(true, true);

    // Decode a prompt on the target, requesting logits/nextn for the last
    // token.
    let tokens = model
        .str_to_token("The quick brown fox", AddBos::Always)
        .expect("failed to tokenize prompt");
    let last_index = i32::try_from(tokens.len()).expect("prompt too long") - 1;

    let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
    for (i, token) in (0_i32..).zip(tokens.iter().copied()) {
        batch
            .add(token, i, &[0], i == last_index)
            .expect("failed to add token to target batch");
    }
    target.decode(&mut batch).expect("target decode failed");

    // The nextn row must have length n_embd and be all-finite.
    let nextn = target
        .get_embeddings_nextn_ith(last_index)
        .expect("target produced no nextn row for the last token");
    assert_eq!(
        nextn.len(),
        n_embd,
        "nextn row length should equal model n_embd"
    );
    assert!(
        nextn.iter().all(|f| f.is_finite()),
        "every nextn value must be finite"
    );
    let nextn_row = nextn.to_vec();

    let last_token = *tokens.last().expect("prompt must have at least one token");

    // Carry (token, nextn row) into the MTP batch and decode on the draft.
    let mut mtp_batch = LlamaMtpBatch::new(1, n_embd);
    mtp_batch
        .add(last_token, &nextn_row, last_index, 0, true)
        .expect("failed to fill the MTP batch");
    draft
        .decode_mtp(&mut mtp_batch)
        .expect("draft (MTP) decode failed");

    // The draft must propose at least one token from the embd batch.
    let proposed: Vec<_> = draft.candidates_ith(last_index).collect();
    assert!(
        !proposed.is_empty(),
        "draft context proposed no tokens from the MTP batch"
    );
}
