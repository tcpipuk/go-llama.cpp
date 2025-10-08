#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

// Model parameters for loading
typedef struct {
    int n_ctx;              // Context size
    int n_batch;            // Batch size
    int n_gpu_layers;       // Number of GPU layers
    int n_threads;          // Number of threads for generation (per token)
    int n_threads_batch;    // Number of threads for batch processing (prompt)
    bool f16_memory;        // Use F16 for memory
    bool mlock;            // Memory lock
    bool mmap;             // Memory mapping
    bool embeddings;       // Enable embeddings
    const char* main_gpu;   // Main GPU
    const char* tensor_split; // Tensor split
} llama_wrapper_model_params;

// Generation parameters
typedef struct {
    const char* prompt;
    int max_tokens;
    int seed;
    const char** stop_words;
    int stop_words_count;
    int n_draft;           // For speculative sampling
    bool debug;
    uintptr_t callback_handle; // Handle to Go callback
    bool enable_prefix_caching; // Enable KV cache reuse for matching prefixes

    // Basic sampling parameters
    float temperature;
    int top_k;
    float top_p;
    float min_p;
    float typ_p;
    float top_n_sigma;
    int min_keep;

    // Repetition penalties
    int penalty_last_n;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;

    // DRY sampling
    float dry_multiplier;
    float dry_base;
    int dry_allowed_length;
    int dry_penalty_last_n;
    const char** dry_sequence_breakers;
    int dry_sequence_breakers_count;

    // Dynamic temperature
    float dynatemp_range;
    float dynatemp_exponent;

    // XTC sampling
    float xtc_probability;
    float xtc_threshold;

    // Mirostat sampling
    int mirostat;
    float mirostat_tau;
    float mirostat_eta;

    // Other parameters
    int n_prev;
    int n_probs;
    bool ignore_eos;
} llama_wrapper_generate_params;

// Callback for streaming tokens
typedef bool (*llama_wrapper_token_callback)(const char* token);

// Logging initialization
void llama_wrapper_init_logging();

// Model management
void* llama_wrapper_model_load(const char* model_path, llama_wrapper_model_params params);
void llama_wrapper_model_free(void* model);

// Context management (kept for API compatibility)
void* llama_wrapper_context_create(void* model, llama_wrapper_model_params params);
void llama_wrapper_context_free(void* ctx);

// Text generation
char* llama_wrapper_generate(void* ctx, llama_wrapper_generate_params params);
char* llama_wrapper_generate_with_tokens(void* ctx, const int* tokens, int n_tokens, int prefix_len, llama_wrapper_generate_params params);

// Speculative generation with draft model
char* llama_wrapper_generate_draft(void* ctx_target, void* ctx_draft, llama_wrapper_generate_params params);
char* llama_wrapper_generate_draft_with_tokens(void* ctx_target, void* ctx_draft, const int* tokens, int n_tokens, int target_prefix_len, int draft_prefix_len, llama_wrapper_generate_params params);

// Tokenization
int llama_wrapper_tokenize(void* ctx, const char* text, int* tokens, int max_tokens);

// Embeddings
int llama_wrapper_embeddings(void* ctx, const char* text, float* embeddings, int max_embeddings);

// Utility functions
void llama_wrapper_free_result(char* result);
const char* llama_wrapper_last_error();
int llama_wrapper_get_cached_token_count(void* ctx);

// Get model's native maximum context length
int llama_wrapper_get_model_context_length(void* model);

#ifdef __cplusplus
}
#endif
