// C-only shim for the pre-norm embedding functions defined in
// llama.cpp/src/llama-ext.h. That header is a C++ staging header (it pulls in
// <map>/std::map via llama_memory_breakdown), so it cannot be fed to bindgen
// directly. Forward-declare only the three C-compatible LLAMA_API pre-norm
// functions here so bindgen can generate FFI for them.
#include "llama.cpp/include/llama.h"

#include <stdint.h>

// Set whether the context outputs pre-norm embeddings or not.
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0.
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits.
LLAMA_API void llama_set_embeddings_pre_norm(struct llama_context * ctx, bool value, bool masked);

// Mirrors llama_get_embeddings, but returns the pre-norm hidden state.
LLAMA_API float * llama_get_embeddings_pre_norm(struct llama_context * ctx);

// Mirrors llama_get_embeddings_ith, but returns the pre-norm hidden state.
LLAMA_API float * llama_get_embeddings_pre_norm_ith(struct llama_context * ctx, int32_t i);
