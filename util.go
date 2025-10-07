package llama

import (
	"fmt"
	"unsafe"
)

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

// Tokenize converts text to tokens.
//
// Tokens are integer IDs representing subword units in the model's vocabulary.
// This method is useful for advanced use cases like manual prompt construction,
// token counting, or analysis. For normal generation, use Generate() which
// handles tokenisation automatically.
//
// Examples:
//
//	// Count tokens in a prompt
//	tokens, _ := model.Tokenize("Hello world")
//	fmt.Printf("Token count: %d\n", len(tokens))
//
//	// Manual generation with tokens
//	tokens, _ := model.Tokenize("Once upon a time")
//	result, _ := model.GenerateWithTokens(tokens)
func (m *Model) Tokenize(text string) ([]int32, error) {
	m.mu.RLock() // Read lock to check closed state
	defer m.mu.RUnlock()

	if m.closed {
		return nil, fmt.Errorf("model is closed")
	}

	// Acquire context from pool
	ctx, err := m.pool.acquire()
	if err != nil {
		return nil, fmt.Errorf("failed to acquire context: %w", err)
	}
	defer m.pool.release(ctx)

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	maxTokens := 8192 // Reasonable upper bound
	tokens := make([]C.int, maxTokens)

	count := C.llama_wrapper_tokenize(ctx.ptr, cText, &tokens[0], C.int(maxTokens))
	if count < 0 {
		return nil, fmt.Errorf("tokenisation failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	result := make([]int32, count)
	for i := 0; i < int(count); i++ {
		result[i] = int32(tokens[i])
	}

	return result, nil
}

// GetCachedTokenCount returns the number of cached tokens in a context (for debugging/metrics).
//
// This method provides insight into prefix caching behaviour, showing how many
// tokens from previous prompts are cached. Useful for diagnostics, performance
// monitoring, or validating prefix caching is working as expected.
//
// Note: This acquires a context from the pool, so the count may vary between
// calls depending on which context is available (each context maintains its
// own cache state).
//
// Example:
//
//	model.Generate("System prompt: You are helpful.\n\nUser: Hello")
//	cached, _ := model.GetCachedTokenCount()
//	fmt.Printf("Cached tokens: %d\n", cached)
//
//	model.Generate("System prompt: You are helpful.\n\nUser: Goodbye")
//	cached, _ = model.GetCachedTokenCount()
//	fmt.Printf("Cached tokens after reuse: %d\n", cached)  // Should be higher
func (m *Model) GetCachedTokenCount() (int, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return 0, fmt.Errorf("model is closed")
	}

	// Acquire context from pool
	ctx, err := m.pool.acquire()
	if err != nil {
		return 0, fmt.Errorf("failed to acquire context: %w", err)
	}
	defer m.pool.release(ctx)

	count := int(C.llama_wrapper_get_cached_token_count(ctx.ptr))
	if count < 0 {
		return 0, fmt.Errorf("failed to get cached token count: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	return count, nil
}

// GetEmbeddings computes embeddings for the given text.
//
// Embeddings are vector representations useful for semantic search, clustering,
// or similarity tasks. The model must be loaded with WithEmbeddings() to use
// this method, otherwise it returns an error. Not all models support embeddings -
// check model documentation.
//
// Example:
//
//	model, _ := llama.LoadModel("model.gguf",
//	    llama.WithEmbeddings(),
//	)
//	defer model.Close()
//
//	emb1, _ := model.GetEmbeddings("Hello world")
//	emb2, _ := model.GetEmbeddings("Hi there")
//	similarity := cosineSimilarity(emb1, emb2)
func (m *Model) GetEmbeddings(text string) ([]float32, error) {
	m.mu.RLock() // Read lock to check closed state
	defer m.mu.RUnlock()

	if m.closed {
		return nil, fmt.Errorf("model is closed")
	}

	// Acquire context from pool
	ctx, err := m.pool.acquire()
	if err != nil {
		return nil, fmt.Errorf("failed to acquire context: %w", err)
	}
	defer m.pool.release(ctx)

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	maxEmbeddings := 4096 // Reasonable upper bound
	embeddings := make([]C.float, maxEmbeddings)

	count := C.llama_wrapper_embeddings(ctx.ptr, cText, &embeddings[0], C.int(maxEmbeddings))
	if count < 0 {
		return nil, fmt.Errorf("embedding generation failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	result := make([]float32, count)
	for i := 0; i < int(count); i++ {
		result[i] = float32(embeddings[i])
	}

	return result, nil
}
