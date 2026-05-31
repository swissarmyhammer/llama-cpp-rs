//! utilities for working with session files

use crate::context::LlamaContext;
use crate::token::LlamaToken;
use std::ffi::{CString, NulError};
use std::path::{Path, PathBuf};

/// Failed to save a Session file
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SaveSessionError {
    /// llama.cpp failed to save the session file
    #[error("Failed to save session file")]
    FailedToSave,

    /// null byte in string
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// failed to convert path to str
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),
}

/// Failed to restore a single sequence's state via [`LlamaContext::state_seq_set_data`].
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SetSeqStateError {
    /// llama.cpp reported `0` bytes read, meaning the state could not be loaded
    /// (e.g. the buffer is truncated, garbage, or incompatible with the context).
    #[error("Failed to set sequence state from the provided bytes")]
    FailedToSet,
}

/// Failed to load a Session file
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LoadSessionError {
    /// llama.cpp failed to load the session file
    #[error("Failed to load session file")]
    FailedToLoad,

    /// null byte in string
    #[error("null byte in string {0}")]
    NullError(#[from] NulError),

    /// failed to convert path to str
    #[error("failed to convert path {0} to str")]
    PathToStrError(PathBuf),

    /// Insufficient max length
    #[error("max_length is not large enough to hold {n_out} (was {max_tokens})")]
    InsufficientMaxLength {
        /// The length of the session file
        n_out: usize,
        /// The maximum length
        max_tokens: usize,
    },
}

impl LlamaContext<'_> {
    /// Save the current session to a file.
    ///
    /// # Parameters
    ///
    /// * `path_session` - The file to save to.
    /// * `tokens` - The tokens to associate the session with. This should be a prefix of a sequence of tokens that the context has processed, so that the relevant KV caches are already filled.
    ///
    /// # Errors
    ///
    /// Fails if the path is not a valid utf8, is not a valid c string, or llama.cpp fails to save the session file.
    pub fn save_session_file(
        &self,
        path_session: impl AsRef<Path>,
        tokens: &[LlamaToken],
    ) -> Result<(), SaveSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or_else(|| SaveSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;

        if unsafe {
            llama_cpp_sys_2::llama_save_session_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens.as_ptr().cast::<llama_cpp_sys_2::llama_token>(),
                tokens.len(),
            )
        } {
            Ok(())
        } else {
            Err(SaveSessionError::FailedToSave)
        }
    }
    /// Load a session file into the current context.
    ///
    /// You still need to pass the returned tokens to the context for inference to work. What this function buys you is that the KV caches are already filled with the relevant data.
    ///
    /// # Parameters
    ///
    /// * `path_session` - The file to load from. It must be a session file from a compatible context, otherwise the function will error.
    /// * `max_tokens` - The maximum token length of the loaded session. If the session was saved with a longer length, the function will error.
    ///
    /// # Errors
    ///
    /// Fails if the path is not a valid utf8, is not a valid c string, or llama.cpp fails to load the session file. (e.g. the file does not exist, is not a session file, etc.)
    pub fn load_session_file(
        &mut self,
        path_session: impl AsRef<Path>,
        max_tokens: usize,
    ) -> Result<Vec<LlamaToken>, LoadSessionError> {
        let path = path_session.as_ref();
        let path = path
            .to_str()
            .ok_or(LoadSessionError::PathToStrError(path.to_path_buf()))?;

        let cstr = CString::new(path)?;
        let mut tokens: Vec<LlamaToken> = Vec::with_capacity(max_tokens);
        let mut n_out = 0;

        // SAFETY: cast is valid as LlamaToken is repr(transparent)
        let tokens_out = tokens.as_mut_ptr().cast::<llama_cpp_sys_2::llama_token>();

        let load_session_success = unsafe {
            llama_cpp_sys_2::llama_load_session_file(
                self.context.as_ptr(),
                cstr.as_ptr(),
                tokens_out,
                max_tokens,
                &raw mut n_out,
            )
        };
        if load_session_success {
            if n_out > max_tokens {
                return Err(LoadSessionError::InsufficientMaxLength { n_out, max_tokens });
            }
            // SAFETY: we checked that n_out <= max_tokens and llama.cpp promises that n_out tokens will be written
            unsafe {
                tokens.set_len(n_out);
            }
            Ok(tokens)
        } else {
            Err(LoadSessionError::FailedToLoad)
        }
    }

    /// Returns the maximum size in bytes of the state (rng, logits, embedding
    /// and `kv_cache`) - will often be smaller after compacting tokens
    #[must_use]
    pub fn get_state_size(&self) -> usize {
        unsafe { llama_cpp_sys_2::llama_get_state_size(self.context.as_ptr()) }
    }

    /// Copies the state to the specified destination address.
    ///
    /// Returns the number of bytes copied
    ///
    /// # Safety
    ///
    /// Destination needs to have allocated enough memory.
    pub unsafe fn copy_state_data(&self, dest: *mut u8) -> usize {
        unsafe { llama_cpp_sys_2::llama_copy_state_data(self.context.as_ptr(), dest) }
    }

    /// Set the state reading from the specified address
    /// Returns the number of bytes read
    ///
    /// # Safety
    ///
    /// help wanted: not entirely sure what the safety requirements are here.
    pub unsafe fn set_state_data(&mut self, src: &[u8]) -> usize {
        unsafe { llama_cpp_sys_2::llama_set_state_data(self.context.as_ptr(), src.as_ptr()) }
    }

    /// Returns the exact size in bytes needed to snapshot the state of a single
    /// sequence (its `kv_cache` entries) via [`Self::state_seq_get_data`].
    #[must_use]
    pub fn state_seq_get_size(&self, seq_id: i32) -> usize {
        unsafe { llama_cpp_sys_2::llama_state_seq_get_size(self.context.as_ptr(), seq_id) }
    }

    /// Snapshot the state of sequence `seq_id` into a freshly allocated `Vec<u8>`.
    ///
    /// The buffer is sized exactly to [`Self::state_seq_get_size`], so callers
    /// cannot under-size it. Pair with [`Self::state_seq_set_data`] to restore.
    #[must_use]
    pub fn state_seq_get_data(&self, seq_id: i32) -> Vec<u8> {
        let size = self.state_seq_get_size(seq_id);
        let mut buf: Vec<u8> = Vec::with_capacity(size);
        // SAFETY: `buf` has capacity for `size` bytes, and llama.cpp writes at
        // most `size` bytes (the value it just reported via state_seq_get_size).
        let written = unsafe {
            llama_cpp_sys_2::llama_state_seq_get_data(
                self.context.as_ptr(),
                buf.as_mut_ptr(),
                size,
                seq_id,
            )
        };
        // SAFETY: llama.cpp wrote `written` (<= size) initialized bytes.
        unsafe {
            buf.set_len(written);
        }
        buf
    }

    /// Restore a sequence state previously snapshotted by
    /// [`Self::state_seq_get_data`] into `dest_seq_id`.
    ///
    /// Returns the number of bytes read on success.
    ///
    /// # Errors
    ///
    /// Returns [`SetSeqStateError::FailedToSet`] if llama.cpp reports `0` bytes
    /// read, which happens when `src` is truncated, corrupt, or incompatible
    /// with this context.
    pub fn state_seq_set_data(
        &mut self,
        src: &[u8],
        dest_seq_id: i32,
    ) -> Result<usize, SetSeqStateError> {
        // SAFETY: `src` is a valid slice of `src.len()` bytes; llama.cpp reads
        // at most `src.len()` bytes from it.
        let read = unsafe {
            llama_cpp_sys_2::llama_state_seq_set_data(
                self.context.as_ptr(),
                src.as_ptr(),
                src.len(),
                dest_seq_id,
            )
        };
        if read == 0 {
            Err(SetSeqStateError::FailedToSet)
        } else {
            Ok(read)
        }
    }
}
