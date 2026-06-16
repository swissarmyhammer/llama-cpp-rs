//! Accessors for the MTP (multi-token-prediction) `nextn` hidden state.
//!
//! The model's MTP / NextN head consumes a per-position hidden-state row that
//! the backend exposes separately from the normal embeddings output. These
//! accessors surface that row from the safe [`LlamaContext`], mirroring the
//! existing `embeddings_ith` / `embeddings_seq_ith` accessors and wrapping the
//! upstream `llama_*_embeddings_nextn` API. (Upstream previously exposed this as
//! a pre-norm hidden state; as of the qwen35 MTP change (#24025) it is the
//! post-norm `nextn` row — same access pattern, renamed surface.)

use std::num::NonZeroI32;
use std::slice;

use crate::context::LlamaContext;
use crate::mtp_batch::LlamaMtpBatch;
use crate::DecodeError;

/// Turn a raw nextn hidden-state pointer returned by the backend into a safe
/// slice, guarding against the null pointer the backend returns when no row was
/// produced.
///
/// Returns `None` when `ptr` is null (nextn disabled, or no decode produced
/// the requested row); otherwise a slice of `n_embd` elements.
///
/// # Safety
///
/// When `ptr` is non-null it must point to at least `n_embd` contiguous `f32`
/// values that stay valid for the chosen lifetime `'a`. Callers bind `'a` to
/// `&self` so the slice cannot outlive the context; the backend invalidates the
/// buffer on the next `decode`.
unsafe fn nextn_slice<'a>(ptr: *const f32, n_embd: usize) -> Option<&'a [f32]> {
    if ptr.is_null() {
        None
    } else {
        Some(slice::from_raw_parts(ptr, n_embd))
    }
}

impl LlamaContext<'_> {
    /// Decodes an embedding-carrying [`LlamaMtpBatch`] for Multi-Token
    /// Prediction draft generation.
    ///
    /// This mirrors [`Self::decode`] but submits a batch that carries both
    /// token ids and per-position nextn embedding rows, as required by the
    /// model's MTP/NextN head. It is intended for a context created with
    /// [`LlamaContextType::Mtp`](crate::context::params::LlamaContextType::Mtp).
    /// After a successful decode, read the proposed draft logits with
    /// [`Self::get_logits_ith`] (or [`Self::candidates_ith`]) for any position
    /// whose `logits` flag was set when it was added to the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode_mtp(&mut self, batch: &mut LlamaMtpBatch) -> Result<(), DecodeError> {
        let result = unsafe { llama_cpp_sys_2::llama_decode(self.context.as_ptr(), batch.inner) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Enable or disable nextn hidden-state output.
    ///
    /// `masked == true` outputs rows only for tokens with `batch.logits != 0`;
    /// `false` outputs all tokens. Wraps `llama_set_embeddings_nextn`.
    pub fn set_embeddings_nextn(&mut self, enabled: bool, masked: bool) {
        unsafe {
            llama_cpp_sys_2::llama_set_embeddings_nextn(self.context.as_ptr(), enabled, masked);
        }
    }

    /// Get the nextn hidden state for output `i` in the current context.
    ///
    /// # Returns
    ///
    /// A slice of length `n_embd` (the context model's embedding dimension), or
    /// `None` if the backend produced no row for `i` — nextn output was not
    /// enabled, or no decode produced this row. Wraps
    /// `llama_get_embeddings_nextn_ith`.
    ///
    /// The returned slice borrows from the context and is invalidated by the
    /// next `decode`.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    #[must_use]
    pub fn get_embeddings_nextn_ith(&self, i: i32) -> Option<&[f32]> {
        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let ptr = llama_cpp_sys_2::llama_get_embeddings_nextn_ith(self.context.as_ptr(), i);
            nextn_slice(ptr, n_embd)
        }
    }

    /// Get the nextn hidden state for the whole batch in the current context.
    ///
    /// # Returns
    ///
    /// A slice of length `n_embd` (the context model's embedding dimension), or
    /// `None` if the backend produced no rows — nextn output was not enabled,
    /// or no decode has run. Wraps `llama_get_embeddings_nextn`.
    ///
    /// The returned slice borrows from the context and is invalidated by the
    /// next `decode`.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    #[must_use]
    pub fn get_embeddings_nextn(&self) -> Option<&[f32]> {
        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let ptr = llama_cpp_sys_2::llama_get_embeddings_nextn(self.context.as_ptr());
            nextn_slice(ptr, n_embd)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::nextn_slice;

    #[test]
    fn null_pointer_yields_none() {
        // The backend returns a null pointer when nextn output was never
        // produced (nextn disabled, or no decode produced this row). The
        // guard must surface that as `None` rather than reading a dangling
        // pointer.
        let slice = unsafe { nextn_slice(std::ptr::null(), 4) };
        assert!(slice.is_none());
    }

    #[test]
    fn non_null_pointer_yields_slice_of_n_embd() {
        let buf = [1.0_f32, 2.0, 3.0, 4.0];
        let slice = unsafe { nextn_slice(buf.as_ptr(), buf.len()) };
        assert_eq!(slice, Some(&buf[..]));
    }
}
