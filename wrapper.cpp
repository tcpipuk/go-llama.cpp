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
    model_params.n_gpu_layers = params.n_gpu_layers;
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

char* llama_wrapper_generate(void* ctx, llama_wrapper_generate_params params) {
    if (!ctx) {
        g_last_error = "Context cannot be null";
        return nullptr;
    }

    auto wrapper = static_cast<llama_wrapper_context_t*>(ctx);

    try {
        // Tokenize the prompt
        std::vector<llama_token> prompt_tokens = common_tokenize(wrapper->ctx, params.prompt, true, true);

        if (prompt_tokens.empty()) {
            g_last_error = "Failed to tokenize prompt";
            return nullptr;
        }

        // Check context size with safety margin
        int available_ctx = llama_n_ctx(wrapper->ctx);
        if (available_ctx <= 0) {
            g_last_error = "Invalid context size";
            return nullptr;
        }
        if ((int)prompt_tokens.size() >= available_ctx - 1) {
            g_last_error = "Prompt too long for context size (need at least 1 token for generation)";
            return nullptr;
        }

        // Create sampling parameters - use the struct directly instead of calling a function
        common_params_sampling sampling_params;
        sampling_params.temp = params.temperature;
        sampling_params.top_p = params.top_p;
        sampling_params.top_k = params.top_k;
        sampling_params.seed = params.seed;

        // Initialize sampler
        common_sampler* sampler = common_sampler_init(wrapper->model, sampling_params);
        if (!sampler) {
            g_last_error = "Failed to initialize sampler";
            return nullptr;
        }

        // Validate generation parameters
        int n_predict = params.max_tokens > 0 ? params.max_tokens : 128;
        if (n_predict <= 0 || n_predict > 8192) {
            common_sampler_free(sampler);
            g_last_error = "Invalid max_tokens value (must be 1-8192)";
            return nullptr;
        }

        // Prepare initial batch with the prompt - following llama.cpp simple.cpp exactly
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

        // Generation loop following llama.cpp/examples/simple/simple.cpp pattern
        std::string result;
        int n_decode = 0;
        llama_token new_token_id;

        // Main generation loop - matches simple.cpp structure exactly
        for (int n_pos = 0; n_pos + batch.n_tokens < (int)prompt_tokens.size() + n_predict; ) {
            // Evaluate the current batch with the transformer model
            // This must happen BEFORE sampling, following llama.cpp pattern
            if (llama_decode(wrapper->ctx, batch) != 0) {
                if (params.debug) {
                    fprintf(stderr, "WARNING: decode failed, stopping generation\n");
                }
                break;
            }

            n_pos += batch.n_tokens;

            // Sample the next token - this happens AFTER decode
            new_token_id = common_sampler_sample(sampler, wrapper->ctx, -1);

            // Check for EOS
            if (llama_vocab_is_eog(llama_model_get_vocab(wrapper->model), new_token_id)) {
                if (params.debug) {
                    fprintf(stderr, "INFO: End of generation token encountered\n");
                }
                break;
            }

            // Convert token to text
            std::string token_str = common_token_to_piece(wrapper->ctx, new_token_id);

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

            // Prepare the next batch with the sampled token - follows simple.cpp exactly
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
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

char* llama_wrapper_generate_draft(void* ctx_target, void* ctx_draft, llama_wrapper_generate_params params) {
    if (!ctx_target || !ctx_draft) {
        g_last_error = "Target and draft contexts cannot be null";
        return nullptr;
    }

    auto wrapper_tgt = static_cast<llama_wrapper_context_t*>(ctx_target);
    auto wrapper_dft = static_cast<llama_wrapper_context_t*>(ctx_draft);

    try {
        // Tokenize the prompt
        std::vector<llama_token> prompt_tokens = common_tokenize(wrapper_tgt->ctx, params.prompt, true, true);

        if (prompt_tokens.empty()) {
            g_last_error = "Failed to tokenize prompt";
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

        // Initialize sampler
        common_sampler* sampler = common_sampler_init(wrapper_tgt->model, sampling_params);
        if (!sampler) {
            common_speculative_free(spec);
            g_last_error = "Failed to initialize sampler";
            return nullptr;
        }

        // Evaluate prompt (all but last token)
        if (prompt_tokens.size() > 1) {
            llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size() - 1);
            if (llama_decode(wrapper_tgt->ctx, batch) != 0) {
                common_sampler_free(sampler);
                common_speculative_free(spec);
                g_last_error = "Failed to decode prompt";
                return nullptr;
            }
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

} // extern "C"
