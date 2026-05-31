//! Safe wrapper around an embedding-carrying `llama_batch` for Multi-Token
//! Prediction (MTP) speculative drafting.
//!
//! The MTP draft hook needs a batch that carries **both** a `token` id and an
//! `embd` (pre-norm hidden) row for every position. The general-purpose
//! [`LlamaBatch`](crate::llama_batch::LlamaBatch) is allocated with
//! `llama_batch_init(n, 0, n_seq_max)` (embd = 0) and only fills token rows, so
//! a dedicated type is used here rather than overloading it.
//!
//! This mirrors the allocation pattern of `common_speculative_impl_draft_mtp`
//! in `llama.cpp`: `llama_batch_init(n_tokens, n_embd, 1)` allocates the `embd`
//! buffer (and `pos`/`n_seq_id`/`seq_id`/`logits`) but leaves `token` null, so
//! the token row buffer is allocated separately.

use crate::llama_batch::BatchAddError;
use crate::token::LlamaToken;
use llama_cpp_sys_2::{llama_batch, llama_batch_free, llama_batch_init, llama_pos, llama_token};

/// A safe wrapper around a `llama_batch` that carries both token ids and
/// per-position embedding rows.
///
/// Each of the `allocated` positions holds an `n_embd`-length row in the
/// `embd` buffer plus a single token id. The `embd`, `pos`, `n_seq_id`,
/// `seq_id`, and `logits` buffers are owned by `llama.cpp` (allocated by
/// `llama_batch_init` and released by `llama_batch_free`). The token buffer is
/// owned by Rust via `token_buf` because `llama_batch_init` does not allocate
/// it when `embd > 0`; the raw `inner.token` pointer is only a borrowed view
/// into `token_buf` and is cleared before `llama_batch_free` runs so the two
/// allocators never cross.
#[derive(Debug)]
pub struct LlamaMtpBatch {
    /// The number of positions the batch was allocated for. They are safe to
    /// write to but not necessarily read from until initialized.
    allocated: usize,
    /// The embedding-row length carried by each position.
    n_embd: usize,
    /// Rust-owned backing storage for the token row buffer. `inner.token`
    /// points into this, so it must outlive every use of `inner` and is freed
    /// (by the Rust allocator) when this field drops. It is never read through
    /// directly — all access goes through `inner.token` — hence the allow.
    #[allow(dead_code)]
    token_buf: Vec<llama_token>,
    /// The positions whose logits were requested, in insertion order.
    pub(crate) initialized_logits: Vec<i32>,
    /// The underlying `llama.cpp` batch. Initialized by
    /// `llama_batch_init(allocated, n_embd, 1)`.
    pub(crate) inner: llama_batch,
}

impl LlamaMtpBatch {
    /// Allocate a batch holding up to `n_tokens` positions, each carrying an
    /// `n_embd`-length embedding row. `n_embd` should be `model.n_embd()`.
    ///
    /// # Panics
    ///
    /// Panics if `n_tokens` does not fit into an [`i32`].
    #[must_use]
    pub fn new(n_tokens: usize, n_embd: usize) -> Self {
        let n_tokens_i32 = i32::try_from(n_tokens).expect("cannot fit n_tokens into a i32");
        let n_embd_i32 = i32::try_from(n_embd).expect("cannot fit n_embd into a i32");

        // `llama_batch_init` with embd > 0 allocates the `embd` buffer and the
        // pos/n_seq_id/seq_id/logits buffers, but leaves `token` null.
        let mut batch = unsafe { llama_batch_init(n_tokens_i32, n_embd_i32, 1) };

        // The MTP batch needs token rows too. `llama.cpp` mallocs `batch.token`
        // here; we instead own the buffer in Rust and point `inner.token` at
        // it. `Drop` nulls `inner.token` before `llama_batch_free` so the C
        // allocator never frees this Rust allocation.
        let mut token_buf = vec![0 as llama_token; n_tokens];
        batch.token = token_buf.as_mut_ptr();

        LlamaMtpBatch {
            allocated: n_tokens,
            n_embd,
            token_buf,
            initialized_logits: vec![],
            inner: batch,
        }
    }

    /// Set the next position in the batch: `token` id, the `n_embd`-length
    /// embedding row, position `pos`, sequence id `seq_id`, and whether logits
    /// are requested.
    ///
    /// # Errors
    ///
    /// - [`BatchAddError::InsufficientSpace`] if the batch is already full.
    /// - [`BatchAddError::EmbdLengthMismatch`] if `embd.len() != n_embd`. No
    ///   bytes are written in this case.
    ///
    /// # Panics
    ///
    /// Panics if `self.inner.n_tokens` does not fit into a [`usize`].
    pub fn add(
        &mut self,
        LlamaToken(id): LlamaToken,
        embd: &[f32],
        pos: llama_pos,
        seq_id: i32,
        logits: bool,
    ) -> Result<(), BatchAddError> {
        if self.allocated
            < usize::try_from(self.n_tokens() + 1).expect("cannot fit n_tokens into a usize")
        {
            return Err(BatchAddError::InsufficientSpace(self.allocated));
        }
        if embd.len() != self.n_embd {
            return Err(BatchAddError::EmbdLengthMismatch {
                expected: self.n_embd,
                actual: embd.len(),
            });
        }

        let offset = self.inner.n_tokens;
        let offset_usize = usize::try_from(offset).expect("cannot fit n_tokens into a usize");
        unsafe {
            // batch.token[k] = id;
            self.inner.token.add(offset_usize).write(id);
            // copy the n_embd-length row into batch.embd[k * n_embd ..]
            let embd_dst = self.inner.embd.add(offset_usize * self.n_embd);
            std::ptr::copy_nonoverlapping(embd.as_ptr(), embd_dst, self.n_embd);
            // batch.pos[k] = pos;
            self.inner.pos.add(offset_usize).write(pos);
            // batch.n_seq_id[k] = 1;
            self.inner.n_seq_id.add(offset_usize).write(1);
            // batch.seq_id[k][0] = seq_id;
            let seq_row = *self.inner.seq_id.add(offset_usize);
            seq_row.write(seq_id);
            // batch.logits[k] = logits;
            self.inner.logits.add(offset_usize).write(i8::from(logits));
        }

        if logits {
            self.initialized_logits.push(offset);
        } else {
            self.initialized_logits.retain(|l| l != &offset);
        }

        self.inner.n_tokens += 1;

        Ok(())
    }

    /// Clear the batch. This does not free the associated memory; it resets the
    /// number of tokens to 0 so the batch can be refilled.
    pub fn clear(&mut self) {
        self.inner.n_tokens = 0;
        self.initialized_logits.clear();
    }

    /// Returns the number of positions currently in the batch.
    #[must_use]
    pub fn n_tokens(&self) -> i32 {
        self.inner.n_tokens
    }
}

impl Drop for LlamaMtpBatch {
    /// Frees the batch. The `llama.cpp`-owned buffers are released by
    /// `llama_batch_free`; the Rust-owned `token_buf` is dropped normally.
    ///
    /// `inner.token` is nulled first so `llama_batch_free` does not `free` the
    /// Rust allocation backing `token_buf`.
    fn drop(&mut self) {
        unsafe {
            self.inner.token = std::ptr::null_mut();
            llama_batch_free(self.inner);
        }
        // `token_buf` drops here, releasing the token buffer with the Rust
        // allocator that allocated it.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read back the token id written at position `i` through the raw buffer.
    unsafe fn read_token(batch: &LlamaMtpBatch, i: usize) -> llama_token {
        *batch.inner.token.add(i)
    }

    /// Read back the `n_embd`-length embedding row written at position `i`.
    unsafe fn read_embd_row(batch: &LlamaMtpBatch, i: usize) -> Vec<f32> {
        let start = batch.inner.embd.add(i * batch.n_embd);
        (0..batch.n_embd).map(|j| *start.add(j)).collect()
    }

    /// Build a sentinel embd row for position `i`: `[i*10, i*10+1, ...]` as
    /// `f32`, using lossless `u16 -> f32` conversions to stay lint-clean.
    fn sentinel_row(i: u16, n_embd: usize) -> Vec<f32> {
        (0..u16::try_from(n_embd).unwrap())
            .map(|j| f32::from(i * 10 + j))
            .collect()
    }

    #[test]
    fn add_fills_buffers_and_counts() {
        let n_embd = 4;
        let k: u16 = 3;
        let mut batch = LlamaMtpBatch::new(usize::from(k), n_embd);
        assert_eq!(batch.n_tokens(), 0);

        for i in 0..k {
            let row = sentinel_row(i, n_embd);
            batch
                .add(
                    LlamaToken(i32::from(i) + 100),
                    &row,
                    llama_pos::from(i),
                    0,
                    true,
                )
                .expect("add within capacity should succeed");
            assert_eq!(batch.n_tokens(), i32::from(i) + 1);
        }

        // Verify the raw token and embd buffers hold exactly what was written.
        for i in 0..k {
            let expected_row = sentinel_row(i, n_embd);
            unsafe {
                assert_eq!(read_token(&batch, usize::from(i)), i32::from(i) + 100);
                assert_eq!(read_embd_row(&batch, usize::from(i)), expected_row);
            }
        }
    }

    #[test]
    fn add_past_capacity_returns_insufficient_space() {
        let n_embd = 2;
        let k: u16 = 2;
        let mut batch = LlamaMtpBatch::new(usize::from(k), n_embd);
        let row = vec![1.0_f32; n_embd];

        for i in 0..k {
            batch
                .add(LlamaToken(i32::from(i)), &row, llama_pos::from(i), 0, false)
                .expect("add within capacity should succeed");
        }

        let err = batch
            .add(LlamaToken(99), &row, llama_pos::from(k), 0, false)
            .expect_err("add past capacity must fail");
        assert_eq!(err, BatchAddError::InsufficientSpace(usize::from(k)));
        assert_eq!(batch.n_tokens(), i32::from(k));
    }

    #[test]
    fn add_with_mismatched_embd_len_errors_without_writing() {
        let n_embd = 4;
        let mut batch = LlamaMtpBatch::new(2, n_embd);

        let short = vec![1.0_f32; n_embd - 1];
        let err = batch
            .add(LlamaToken(1), &short, 0, 0, false)
            .expect_err("short embd row must error");
        assert_eq!(
            err,
            BatchAddError::EmbdLengthMismatch {
                expected: n_embd,
                actual: n_embd - 1,
            }
        );

        let long = vec![1.0_f32; n_embd + 1];
        let err = batch
            .add(LlamaToken(1), &long, 0, 0, false)
            .expect_err("long embd row must error");
        assert_eq!(
            err,
            BatchAddError::EmbdLengthMismatch {
                expected: n_embd,
                actual: n_embd + 1,
            }
        );

        // No position was consumed on either failure.
        assert_eq!(batch.n_tokens(), 0);
    }

    #[test]
    fn clear_resets_token_count() {
        let n_embd = 2;
        let mut batch = LlamaMtpBatch::new(2, n_embd);
        let row = vec![0.5_f32; n_embd];
        batch.add(LlamaToken(1), &row, 0, 0, true).unwrap();
        assert_eq!(batch.n_tokens(), 1);
        assert!(!batch.initialized_logits.is_empty());

        batch.clear();
        assert_eq!(batch.n_tokens(), 0);
        assert!(batch.initialized_logits.is_empty());
    }

    #[test]
    fn new_then_drop_is_clean() {
        // Exercise a single new/drop cycle; Drop must free exactly what was
        // allocated (llama.cpp buffers via llama_batch_free, token_buf via Rust).
        let batch = LlamaMtpBatch::new(8, 16);
        drop(batch);
    }
}
