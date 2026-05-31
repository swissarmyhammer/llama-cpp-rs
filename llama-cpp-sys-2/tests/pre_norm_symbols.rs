//! Compile-time proof that the `llama-ext.h` pre-norm embedding functions are
//! bound in the generated FFI.
//!
//! Each assignment coerces the generated symbol to a typed
//! `unsafe extern "C" fn` pointer. If any binding is missing or has the wrong
//! signature, this test fails to compile — and since it references the symbols,
//! linking proves they resolve against libllama.

use llama_cpp_sys_2::{llama_context, llama_get_embeddings_pre_norm};
use llama_cpp_sys_2::{llama_get_embeddings_pre_norm_ith, llama_set_embeddings_pre_norm};

#[test]
fn pre_norm_symbols_are_bound() {
    let _set: unsafe extern "C" fn(*mut llama_context, bool, bool) = llama_set_embeddings_pre_norm;
    let _get: unsafe extern "C" fn(*mut llama_context) -> *mut f32 = llama_get_embeddings_pre_norm;
    let _get_ith: unsafe extern "C" fn(*mut llama_context, i32) -> *mut f32 =
        llama_get_embeddings_pre_norm_ith;
}
