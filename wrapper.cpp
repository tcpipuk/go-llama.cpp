#include "wrapper.h"
#include "llama.cpp/include/llama.h"
#include "llama.cpp/ggml/include/ggml.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/sampling.h"
#include "llama.cpp/common/speculative.h"

#include <string>
#include <vector>
#include <memory>
#include <cstring>

// Global error handling
static std::string g_last_error;

extern "C" {

// Forward declaration of Go callback function
extern bool goTokenCallback(uintptr_t handle, const char* token);

// Separate wrappers for model and context
struct llama_wrapper_model_t {
    llama_model* model;
};

struct llama_wrapper_context_t {
    llama_context* ctx;
    llama_model* model;  // Reference to parent model
    std::vector<int> cached_tokens;  // Cache for prefix matching optimisation
};

const char* llama_wrapper_last_error() {
    return g_last_error.c_str();
}

void llama_wrapper_free_result(char* result) {
    if (result) {
        free(result);
    }
}

// Convert our params to llama.cpp model params
static struct llama_model_params convert_model_params(llama_wrapper_model_params params) {
    struct llama_model_params model_params = llama_model_default_params();

    // Only set n_gpu_layers if not -1 (which means "use default/all layers")
    // llama.cpp default is 999 which effectively means all layers
    if (params.n_gpu_layers != -1) {
        model_params.n_gpu_layers = params.n_gpu_layers;
    }

    model_params.main_gpu = params.main_gpu ? atoi(params.main_gpu) : 0;
    model_params.use_mmap = params.mmap;
    model_params.use_mlock = params.mlock;

    return model_params;
}

// Convert our params to llama.cpp context params
static struct llama_context_params convert_context_params(llama_wrapper_model_params params) {
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx > 0 ? params.n_ctx : 2048;
    ctx_params.n_batch = params.n_batch > 0 ? params.n_batch : 512;
    ctx_params.n_threads = params.n_threads > 0 ? params.n_threads : 4;
    ctx_params.n_threads_batch = params.n_threads_batch > 0 ? params.n_threads_batch : ctx_params.n_threads;
    ctx_params.embeddings = params.embeddings;
    return ctx_params;
}

void* llama_wrapper_model_load(const char* model_path, llama_wrapper_model_params params) {
    if (!model_path) {
        g_last_error = "Model path cannot be null";
        return nullptr;
    }

    try {
        // Initialize llama backend
        llama_backend_init();

        // Load model (weights only)
        auto model_params = convert_model_params(params);
        llama_model* model = llama_model_load_from_file(model_path, model_params);
        if (!model) {
            g_last_error = "Failed to load model from: " + std::string(model_path);
            return nullptr;
        }

        // Create wrapper (model only, no context)
        auto wrapper = new llama_wrapper_model_t();
        wrapper->model = model;

        return wrapper;
    } catch (const std::exception& e) {
        g_last_error = "Exception loading model: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_model_free(void* model) {
    if (!model) return;

    auto wrapper = static_cast<llama_wrapper_model_t*>(model);
    if (wrapper->model) {
        llama_model_free(wrapper->model);
        wrapper->model = nullptr;  // Prevent double-free
    }
    delete wrapper;
}

void* llama_wrapper_context_create(void* model, llama_wrapper_model_params params) {
    if (!model) {
        g_last_error = "Model cannot be null";
        return nullptr;
    }

    try {
        auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

        // Create context from model
        auto ctx_params = convert_context_params(params);
        llama_context* ctx = llama_init_from_model(model_wrapper->model, ctx_params);
        if (!ctx) {
            g_last_error = "Failed to create context";
            return nullptr;
        }

        // Create context wrapper
        auto ctx_wrapper = new llama_wrapper_context_t();
        ctx_wrapper->ctx = ctx;
        ctx_wrapper->model = model_wrapper->model;  // Keep reference to parent model

        return ctx_wrapper;
    } catch (const std::exception& e) {
        g_last_error = "Exception creating context: " + std::string(e.what());
        return nullptr;
    }
}

void llama_wrapper_context_free(void* ctx) {
    if (!ctx) return;

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    if (wrapper->ctx) {
        llama_free(wrapper->ctx);
        wrapper->ctx = nullptr;  // Prevent double-free
    }
    delete wrapper;
}

// Get model's native maximum context length from GGUF metadata
int llama_wrapper_get_model_context_length(void* model) {
    if (!model) {
        return 32768;  // Fallback if model is null
    }

    auto model_wrapper = static_cast<llama_wrapper_model_t*>(model);

    // Query model's native context length from GGUF metadata
    int n_ctx_train = llama_model_n_ctx_train(model_wrapper->model);

    // Return model's training context, or reasonable fallback
    return (n_ctx_train > 0) ? n_ctx_train : 32768;
}

// Helper function to find common prefix length between two token vectors
static int findCommonPrefix(const std::vector<int>& a, const std::vector<int>& b) {
    int commonLen = 0;
    size_t minLen = std::min(a.size(), b.size());
    for (size_t i = 0; i < minLen; i++) {
        if (a[i] != b[i]) {
            break;
        }
        commonLen++;
    }
    return commonLen;
}

char* llama_wrapper_generate_with_tokens(void* ctx, const int* tokens, int n_tokens, int prefix_len, llama_wrapper_generate_params params) {
    if (!ctx || !tokens) {
        g_last_error = "Context and tokens cannot be null";
        return nullptr;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        // Convert C tokens to vector
        std::vector<llama_token> prompt_tokens(tokens, tokens + n_tokens);

        if (prompt_tokens.empty()) {
            g_last_error = "Token array is empty";
            return nullptr;
        }

        // Check context size with safety margin BEFORE manipulating KV cache
        int available_ctx = llama_n_ctx(wrapper->ctx);
        if (available_ctx <= 0) {
            g_last_error = "Invalid context size";
            return nullptr;
        }
        // Check if prompt fits with room for at least a few generated tokens
        int tokens_needed = (int)prompt_tokens.size() + params.max_tokens;
        if (tokens_needed > available_ctx) {
            char err_msg[256];
            snprintf(err_msg, sizeof(err_msg),
                    "Prompt too long for context size: need %d tokens (%d prompt + %d generation) but context is only %d tokens",
                    tokens_needed, (int)prompt_tokens.size(), params.max_tokens > 0 ? params.max_tokens : 128, available_ctx);
            g_last_error = err_msg;
            return nullptr;
        }
        if ((int)prompt_tokens.size() >= available_ctx - 1) {
            g_last_error = "Prompt too long for context size (need at least 1 token for generation)";
            return nullptr;
        }

        // Clear KV cache from divergence point onwards
        // For full cache hits, we'll refresh the last prompt token, so clear from prefix_len - 1
        // For partial matches, clear from prefix_len as usual
        int clear_from = (prefix_len == n_tokens && n_tokens > 0) ? prefix_len - 1 : prefix_len;
        // Only clear if clear_from is valid and within context bounds
        if (clear_from >= 0 && clear_from < available_ctx) {
            llama_memory_seq_rm(llama_get_memory(wrapper->ctx), 0, clear_from, -1);
        }

        // Create sampling parameters - use the struct directly instead of calling a function
        common_params_sampling sampling_params;
        sampling_params.temp = params.temperature;
        sampling_params.top_p = params.top_p;
        sampling_params.top_k = params.top_k;
        sampling_params.seed = params.seed;

        // Initialise sampler
        common_sampler* sampler = common_sampler_init(wrapper->model, sampling_params);
        if (!sampler) {
            g_last_error = "Failed to initialise sampler";
            return nullptr;
        }

        // Validate generation parameters
        // Reject negative max_tokens (0 is allowed and means "use default")
        if (params.max_tokens < 0) {
            common_sampler_free(sampler);
            g_last_error = "Invalid max_tokens value (must be >= 0)";
            return nullptr;
        }
        int n_predict = params.max_tokens > 0 ? params.max_tokens : 128;

        // After clearing cache from prefix_len onwards, cache ends at prefix_len - 1
        // Next position to use is prefix_len
        int n_past = prefix_len;

        // Process prompt tokens from prefix_len onwards using explicit positions
        if (prefix_len < n_tokens) {
            int tokens_to_process = n_tokens - prefix_len;
            llama_batch batch = llama_batch_init(std::max(tokens_to_process, 512), 0, 1);
            common_batch_clear(batch);

            // Add tokens with explicit positions starting from prefix_len
            for (int i = 0; i < tokens_to_process; i++) {
                int token_idx = prefix_len + i;
                bool needs_logits = (i == tokens_to_process - 1);  // Only last token needs logits
                common_batch_add(batch, prompt_tokens[token_idx], prefix_len + i, { 0 }, needs_logits);
            }

            if (llama_decode(wrapper->ctx, batch) != 0) {
                if (params.debug) {
                    fprintf(stderr, "WARNING: prompt decode failed\n");
                }
                llama_batch_free(batch);
                common_sampler_free(sampler);
                g_last_error = "Failed to decode prompt";
                return nullptr;
            }

            llama_batch_free(batch);
            n_past = n_tokens;  // Position now at end of prompt
        } else if (prefix_len == n_tokens && n_tokens > 0) {
            // Full cache hit - refresh last token's logits to ensure determinism
            // This is critical: without this, we sample from stale logits from the previous generation
            // The last prompt token is at position n_tokens - 1 (0-indexed positions)
            llama_batch batch = llama_batch_init(512, 0, 1);
            common_batch_clear(batch);
            common_batch_add(batch, prompt_tokens[n_tokens - 1], n_tokens - 1, { 0 }, true);

            if (llama_decode(wrapper->ctx, batch) != 0) {
                if (params.debug) {
                    fprintf(stderr, "WARNING: logit refresh failed\n");
                }
                llama_batch_free(batch);
                common_sampler_free(sampler);
                g_last_error = "Failed to refresh logits for cached prompt";
                return nullptr;
            }

            llama_batch_free(batch);
            n_past = n_tokens;  // Set position to end of prompt for generation
        }
        // If n_tokens == 0, nothing to decode

        // Generation loop - follows simple.cpp pattern
        std::string result;
        int n_decode = 0;

        if (params.debug) {
            fprintf(stderr, "DEBUG: Starting generation loop, n_predict=%d, n_past=%d\n", n_predict, n_past);
        }

        // Main generation loop - decode first, then sample
        for (int n_gen = 0; n_gen < n_predict; n_gen++) {
            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: First iteration, about to sample\n");
            }

            // Sample the next token (using logits from previous decode or prompt)
            llama_token new_token_id = common_sampler_sample(sampler, wrapper->ctx, -1);

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Sampled token: %d\n", new_token_id);
            }

            // Check for EOS
            if (llama_vocab_is_eog(llama_model_get_vocab(wrapper->model), new_token_id)) {
                if (params.debug) {
                    fprintf(stderr, "INFO: End of generation token encountered\n");
                }
                break;
            }

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: About to convert token to text\n");
            }

            // Convert token to text
            std::string token_str = common_token_to_piece(wrapper->ctx, new_token_id);

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Token text: '%s'\n", token_str.c_str());
            }

            // Call callback if provided
            if (params.callback_handle != 0) {
                if (!goTokenCallback(params.callback_handle, token_str.c_str())) {
                    if (params.debug) {
                        fprintf(stderr, "INFO: Generation stopped by callback\n");
                    }
                    break;
                }
            }

            result += token_str;

            // Check stop words
            for (int j = 0; j < params.stop_words_count; j++) {
                if (result.find(params.stop_words[j]) != std::string::npos) {
                    if (params.debug) {
                        fprintf(stderr, "INFO: Stop word found, ending generation\n");
                    }
                    goto generation_done;
                }
            }

            if (params.debug && n_gen == 0) {
                // Query actual cache state before decode
                int cache_pos = llama_memory_seq_pos_max(llama_get_memory(wrapper->ctx), 0);
                fprintf(stderr, "DEBUG: About to decode token, n_past=%d, cache_pos_max=%d\n", n_past, cache_pos);
            }

            // Decode the sampled token to get logits for next iteration
            // Allocate enough space for the batch (minimum 512 tokens as per llama.cpp examples)
            llama_batch gen_batch = llama_batch_init(512, 0, 1);
            common_batch_clear(gen_batch);
            common_batch_add(gen_batch, new_token_id, n_past, { 0 }, true);

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Batch token=%d, pos=%d, n_tokens=%d\n", new_token_id, n_past, gen_batch.n_tokens);
            }

            // Increment position for next iteration
            n_past++;

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Batch prepared, calling llama_decode\n");
            }

            if (llama_decode(wrapper->ctx, gen_batch) != 0) {
                if (params.debug) {
                    fprintf(stderr, "WARNING: decode failed, stopping generation\n");
                }
                llama_batch_free(gen_batch);
                break;
            }

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: Decode succeeded, freeing batch\n");
            }

            llama_batch_free(gen_batch);
            n_decode += 1;

            if (params.debug && n_gen == 0) {
                fprintf(stderr, "DEBUG: First iteration complete\n");
            }
        }

generation_done:
        common_sampler_free(sampler);

        // Return allocated string (caller must free)
        char* c_result = (char*)malloc(result.length() + 1);
        if (c_result) {
            memcpy(c_result, result.c_str(), result.length());
            c_result[result.length()] = '\0';
        } else {
            g_last_error = "Failed to allocate memory for result";
        }
        return c_result;

    } catch (const std::exception& e) {
        g_last_error = "Exception during generation: " + std::string(e.what());
        return nullptr;
    }
}

// Simple wrapper that tokenises the prompt and handles prefix caching automatically
char* llama_wrapper_generate(void* ctx, llama_wrapper_generate_params params) {
    if (!ctx) {
        g_last_error = "Context cannot be null";
        return nullptr;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        // Tokenise the prompt
        std::vector<llama_token> prompt_tokens = common_tokenize(wrapper->ctx, params.prompt, true, true);

        if (prompt_tokens.empty()) {
            g_last_error = "Failed to tokenise prompt";
            return nullptr;
        }

        // Convert to int vector for comparison
        std::vector<int> tokens_int(prompt_tokens.begin(), prompt_tokens.end());

        // Find common prefix with cached tokens (only if prefix caching enabled)
        int prefix_len = params.enable_prefix_caching
            ? findCommonPrefix(wrapper->cached_tokens, tokens_int)
            : 0;

        // Update cache to new token sequence (only if prefix caching enabled)
        if (params.enable_prefix_caching) {
            wrapper->cached_tokens = tokens_int;
        } else {
            wrapper->cached_tokens.clear();  // Ensure cache is empty when disabled
        }

        // Call token-based generation with prefix caching
        return llama_wrapper_generate_with_tokens(ctx, tokens_int.data(), tokens_int.size(), prefix_len, params);

    } catch (const std::exception& e) {
        g_last_error = "Exception during generation: " + std::string(e.what());
        return nullptr;
    }
}

char* llama_wrapper_generate_draft_with_tokens(void* ctx_target, void* ctx_draft, const int* tokens, int n_tokens, int target_prefix_len, int draft_prefix_len, llama_wrapper_generate_params params) {
    if (!ctx_target || !ctx_draft || !tokens) {
        g_last_error = "Target, draft contexts and tokens cannot be null";
        return nullptr;
    }

    auto wrapper_tgt = static_cast<llama_wrapper_context_t*>(ctx_target);
    auto wrapper_dft = static_cast<llama_wrapper_context_t*>(ctx_draft);

    try {
        // Clear KV caches from divergence points
        // Sequence ID 0 is the default sequence for single-sequence inference
        // For speculative generation with full cache hits, we need to refresh the second-to-last token
        // (since we decode all but last token), so clear from that position
        int target_clear_from = (target_prefix_len == n_tokens && n_tokens > 1) ? n_tokens - 2 : target_prefix_len;
        int draft_clear_from = (draft_prefix_len == n_tokens && n_tokens > 1) ? n_tokens - 2 : draft_prefix_len;
        llama_memory_seq_rm(llama_get_memory(wrapper_tgt->ctx), 0, target_clear_from, -1);
        llama_memory_seq_rm(llama_get_memory(wrapper_dft->ctx), 0, draft_clear_from, -1);

        // Convert C tokens to vector
        std::vector<llama_token> prompt_tokens(tokens, tokens + n_tokens);

        if (prompt_tokens.empty()) {
            g_last_error = "Token array is empty";
            return nullptr;
        }

        // Initialize speculative sampling
        common_speculative* spec = common_speculative_init(wrapper_tgt->ctx, wrapper_dft->ctx);
        if (!spec) {
            g_last_error = "Failed to initialize speculative sampling";
            return nullptr;
        }

        // Set up parameters
        common_speculative_params spec_params;
        spec_params.n_draft = params.n_draft > 0 ? params.n_draft : 16;
        spec_params.p_min = 0.75f;

        // Create sampling parameters
        common_params_sampling sampling_params;
        sampling_params.temp = params.temperature;
        sampling_params.top_p = params.top_p;
        sampling_params.top_k = params.top_k;
        sampling_params.seed = params.seed;

        // Initialise sampler
        common_sampler* sampler = common_sampler_init(wrapper_tgt->model, sampling_params);
        if (!sampler) {
            common_speculative_free(spec);
            g_last_error = "Failed to initialise sampler";
            return nullptr;
        }

        // Evaluate prompt (all but last token), but only process tokens after the target prefix
        // If target_prefix_len is at or past the last token, we don't need to decode anything
        if (prompt_tokens.size() > 1 && target_prefix_len < (int)prompt_tokens.size() - 1) {
            // Process tokens from target_prefix_len to size - 1
            int tokens_to_process = prompt_tokens.size() - 1 - target_prefix_len;
            llama_batch batch = llama_batch_init(std::max(tokens_to_process, 512), 0, 1);
            common_batch_clear(batch);

            // Add uncached tokens with correct positions
            for (int i = 0; i < tokens_to_process; i++) {
                int token_idx = target_prefix_len + i;
                bool needs_logits = (i == tokens_to_process - 1); // Only last token needs logits
                common_batch_add(batch, prompt_tokens[token_idx], token_idx, { 0 }, needs_logits);
            }

            if (llama_decode(wrapper_tgt->ctx, batch) != 0) {
                llama_batch_free(batch);
                common_sampler_free(sampler);
                common_speculative_free(spec);
                g_last_error = "Failed to decode prompt";
                return nullptr;
            }
            llama_batch_free(batch);
        } else if (target_prefix_len == (int)prompt_tokens.size() && prompt_tokens.size() > 1) {
            // Full cache hit - refresh the second-to-last token to ensure determinism
            // This matches the pattern where we decode all but the last token
            llama_batch batch = llama_batch_init(512, 0, 1);
            common_batch_clear(batch);
            common_batch_add(batch, prompt_tokens[prompt_tokens.size() - 2], prompt_tokens.size() - 2, { 0 }, true);

            if (llama_decode(wrapper_tgt->ctx, batch) != 0) {
                if (params.debug) {
                    fprintf(stderr, "WARNING: speculative prompt logit refresh failed\n");
                }
                llama_batch_free(batch);
                common_sampler_free(sampler);
                common_speculative_free(spec);
                g_last_error = "Failed to refresh logits for cached speculative prompt";
                return nullptr;
            }
            llama_batch_free(batch);
        }

        // Generation variables
        std::string result;
        llama_token last_token = prompt_tokens.back();
        llama_tokens prompt_tgt(prompt_tokens.begin(), prompt_tokens.end() - 1);
        int n_past = prompt_tokens.size() - 1;
        int n_predict = params.max_tokens > 0 ? params.max_tokens : 128;

        llama_batch batch_tgt = llama_batch_init(llama_n_batch(wrapper_tgt->ctx), 0, 1);

        // Generation loop
        while (result.length() < (size_t)n_predict) {
            // Generate draft tokens
            llama_tokens draft = common_speculative_gen_draft(spec, spec_params, prompt_tgt, last_token);

            // Prepare batch with last token and draft
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, last_token, n_past, { 0 }, true);

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i + 1, { 0 }, true);
            }

            // Evaluate on target model
            if (llama_decode(wrapper_tgt->ctx, batch_tgt) != 0) {
                if (params.debug) {
                    fprintf(stderr, "WARNING: target decode failed, stopping\n");
                }
                break;
            }

            // Sample and accept tokens
            const auto ids = common_sampler_sample_and_accept_n(sampler, wrapper_tgt->ctx, draft);

            if (ids.empty()) {
                break;
            }

            // Update position tracking - FIXED: only increment by actual accepted tokens
            n_past += ids.size();

            // Process accepted tokens
            for (size_t i = 0; i < ids.size(); ++i) {
                const llama_token id = ids[i];

                // Check for EOS
                if (llama_vocab_is_eog(llama_model_get_vocab(wrapper_tgt->model), id)) {
                    goto speculative_done;
                }

                const std::string token_str = common_token_to_piece(wrapper_tgt->ctx, id);

                // Call callback if provided
                if (params.callback_handle != 0) {
                    if (!goTokenCallback(params.callback_handle, token_str.c_str())) {
                        goto speculative_done;
                    }
                }

                result += token_str;
                prompt_tgt.push_back(id);

                // Check stop words
                for (int j = 0; j < params.stop_words_count; j++) {
                    if (result.find(params.stop_words[j]) != std::string::npos) {
                        goto speculative_done;
                    }
                }
            }

            // Update last token for next iteration
            if (!ids.empty()) {
                last_token = ids.back();
            }
        }

speculative_done:
        llama_batch_free(batch_tgt);
        common_sampler_free(sampler);
        common_speculative_free(spec);

        // Return allocated string
        char* c_result = (char*)malloc(result.length() + 1);
        if (c_result) {
            memcpy(c_result, result.c_str(), result.length());
            c_result[result.length()] = '\0';
        } else {
            g_last_error = "Failed to allocate memory for result";
        }
        return c_result;

    } catch (const std::exception& e) {
        g_last_error = "Exception during speculative generation: " + std::string(e.what());
        return nullptr;
    }
}

// Simple wrapper that tokenises the prompt and handles prefix caching automatically for both models
char* llama_wrapper_generate_draft(void* ctx_target, void* ctx_draft, llama_wrapper_generate_params params) {
    if (!ctx_target || !ctx_draft) {
        g_last_error = "Target and draft contexts cannot be null";
        return nullptr;
    }

    auto wrapper_tgt = static_cast<llama_wrapper_context_t*>(ctx_target);
    auto wrapper_dft = static_cast<llama_wrapper_context_t*>(ctx_draft);

    try {
        // Tokenise the prompt
        std::vector<llama_token> prompt_tokens = common_tokenize(wrapper_tgt->ctx, params.prompt, true, true);

        if (prompt_tokens.empty()) {
            g_last_error = "Failed to tokenise prompt";
            return nullptr;
        }

        // Convert to int vector for comparison
        std::vector<int> tokens_int(prompt_tokens.begin(), prompt_tokens.end());

        // Find common prefix for both contexts (only if prefix caching enabled)
        int target_prefix_len = params.enable_prefix_caching
            ? findCommonPrefix(wrapper_tgt->cached_tokens, tokens_int)
            : 0;
        int draft_prefix_len = params.enable_prefix_caching
            ? findCommonPrefix(wrapper_dft->cached_tokens, tokens_int)
            : 0;

        // Update both caches to new token sequence (only if prefix caching enabled)
        if (params.enable_prefix_caching) {
            wrapper_tgt->cached_tokens = tokens_int;
            wrapper_dft->cached_tokens = tokens_int;
        } else {
            wrapper_tgt->cached_tokens.clear();  // Ensure cache is empty when disabled
            wrapper_dft->cached_tokens.clear();
        }

        // Call token-based speculative generation with prefix caching
        return llama_wrapper_generate_draft_with_tokens(ctx_target, ctx_draft, tokens_int.data(), tokens_int.size(), target_prefix_len, draft_prefix_len, params);

    } catch (const std::exception& e) {
        g_last_error = "Exception during speculative generation: " + std::string(e.what());
        return nullptr;
    }
}

int llama_wrapper_tokenize(void* ctx, const char* text, int* tokens, int max_tokens) {
    if (!ctx || !text || !tokens) {
        g_last_error = "Invalid parameters for tokenization";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        std::vector<llama_token> token_vec = common_tokenize(wrapper->ctx, text, true, true);

        int count = std::min((int)token_vec.size(), max_tokens);
        for (int i = 0; i < count; i++) {
            tokens[i] = token_vec[i];
        }

        return count;
    } catch (const std::exception& e) {
        g_last_error = "Exception during tokenization: " + std::string(e.what());
        return -1;
    }
}

int llama_wrapper_embeddings(void* ctx, const char* text, float* embeddings, int max_embeddings) {
    if (!ctx || !text || !embeddings) {
        g_last_error = "Invalid parameters for embeddings";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        // Clear KV cache to ensure clean state
        llama_memory_seq_rm(llama_get_memory(wrapper->ctx), 0, -1, -1);

        // Tokenize text
        std::vector<llama_token> tokens = common_tokenize(wrapper->ctx, text, true, true);

        if (tokens.empty()) {
            g_last_error = "Failed to tokenize text for embeddings";
            return -1;
        }

        // Evaluate tokens
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(wrapper->ctx, batch) != 0) {
            g_last_error = "Failed to decode tokens for embeddings";
            return -1;
        }

        // Get embeddings using the correct function call
        const float* embd = llama_get_embeddings(wrapper->ctx);
        if (!embd) {
            g_last_error = "Failed to get embeddings from context";
            return -1;
        }

        // Copy embeddings
        int n_embd = llama_model_n_embd(wrapper->model);
        int count = std::min(n_embd, max_embeddings);

        memcpy(embeddings, embd, count * sizeof(float));

        return count;
    } catch (const std::exception& e) {
        g_last_error = "Exception during embedding generation: " + std::string(e.what());
        return -1;
    }
}

int llama_wrapper_get_cached_token_count(void* ctx) {
    if (!ctx) {
        g_last_error = "Context cannot be null";
        return -1;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);
    return static_cast<int>(wrapper->cached_tokens.size());
}

} // extern "C"
