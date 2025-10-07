package llama

import (
	"fmt"
	"runtime/cgo"
	"unsafe"
)

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

// GenerateWithTokens generates text from pre-tokenised input (advanced API).
//
// This method allows direct control over tokenisation and is useful for advanced
// use cases requiring manual token manipulation or optimisation. For most scenarios,
// use Generate() which handles tokenisation automatically.
//
// The method accepts pre-computed token IDs, bypassing the normal text-to-token
// conversion step. This can be useful for caching tokenised prompts, implementing
// custom tokenisation logic, or working with tokens from external sources.
//
// Example:
//
//	// Tokenise once, reuse multiple times
//	tokens, _ := model.Tokenize("System prompt: You are helpful.\n\n")
//
//	// Generate with different suffixes
//	result1, _ := model.GenerateWithTokens(append(tokens, userTokens1...))
//	result2, _ := model.GenerateWithTokens(append(tokens, userTokens2...))
func (m *Model) GenerateWithTokens(tokens []int32, opts ...GenerateOption) (string, error) {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return m.generateWithTokensAndConfig(tokens, config, nil)
}

// GenerateWithTokensStream generates text with streaming output via callback (advanced API).
//
// This is the streaming variant of GenerateWithTokens. For most use cases,
// prefer GenerateStream() which handles tokenisation automatically.
//
// Example:
//
//	tokens, _ := model.Tokenize("Once upon a time")
//	err := model.GenerateWithTokensStream(tokens,
//	    func(token string) bool {
//	        fmt.Print(token)
//	        return true
//	    },
//	)
func (m *Model) GenerateWithTokensStream(tokens []int32, callback func(token string) bool, opts ...GenerateOption) error {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	_, err := m.generateWithTokensAndConfig(tokens, config, callback)
	return err
}

// Helper function for token-based generation with manual prefix control
func (m *Model) generateWithTokensAndConfig(tokens []int32, config generateConfig, callback func(string) bool) (string, error) {
	// Read lock to check closed state
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.closed {
		return "", fmt.Errorf("model is closed")
	}

	// Acquire context from pool
	ctx, err := m.pool.acquire()
	if err != nil {
		return "", fmt.Errorf("failed to acquire context: %w", err)
	}
	defer m.pool.release(ctx)

	// Get current cached token count from C++
	cachedCount := int(C.llama_wrapper_get_cached_token_count(ctx.ptr))
	if cachedCount < 0 {
		return "", fmt.Errorf("failed to get cached token count: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	// Find common prefix by checking cached tokens from C++
	// We need to fetch the cached tokens to compare
	// For now, we'll let C++ handle this by always using prefix_len=0
	// Advanced users can retrieve tokens via Tokenize and manage themselves
	prefixLen := 0 // No prefix optimisation for manual token input

	// Convert stop words to C array
	var cStopWords **C.char
	var stopWordsCount C.int

	if len(config.stopWords) > 0 {
		stopWordsCount = C.int(len(config.stopWords))
		cStopWordsArray := make([]*C.char, len(config.stopWords))
		for i, word := range config.stopWords {
			cStopWordsArray[i] = C.CString(word)
		}
		defer func() {
			for _, ptr := range cStopWordsArray {
				C.free(unsafe.Pointer(ptr))
			}
		}()
		cStopWords = (**C.char)(unsafe.Pointer(&cStopWordsArray[0]))
	}

	// Set up callback handle if provided
	var handle cgo.Handle
	var callbackHandle C.uintptr_t
	if callback != nil {
		handle = cgo.NewHandle(callback)
		callbackHandle = C.uintptr_t(handle)
	}

	// Convert tokens to C array
	cTokens := make([]C.int, len(tokens))
	for i, tok := range tokens {
		cTokens[i] = C.int(tok)
	}

	params := C.llama_wrapper_generate_params{
		prompt:                nil, // Not used in token-based generation
		max_tokens:            C.int(config.maxTokens),
		temperature:           C.float(config.temperature),
		top_p:                 C.float(config.topP),
		top_k:                 C.int(config.topK),
		seed:                  C.int(config.seed),
		stop_words:            cStopWords,
		stop_words_count:      stopWordsCount,
		debug:                 C.bool(config.debug),
		callback_handle:       callbackHandle,
		enable_prefix_caching: C.bool(config.prefixCaching),
	}

	// Call token-based generation
	var result *C.char
	if len(cTokens) > 0 {
		result = C.llama_wrapper_generate_with_tokens(ctx.ptr, &cTokens[0], C.int(len(tokens)), C.int(prefixLen), params)
	} else {
		result = C.llama_wrapper_generate_with_tokens(ctx.ptr, nil, 0, C.int(prefixLen), params)
	}

	// Clean up handle if it was created
	if callback != nil {
		handle.Delete()
	}

	if result == nil {
		return "", fmt.Errorf("generation failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	defer C.llama_wrapper_free_result(result)

	return C.GoString(result), nil
}
