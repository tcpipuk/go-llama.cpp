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

//export goTokenCallback
func goTokenCallback(handle C.uintptr_t, token *C.char) C.bool {
	h := cgo.Handle(handle)
	callback := h.Value().(func(string) bool)
	return C.bool(callback(C.GoString(token)))
}

// findCommonPrefix returns length of common prefix between two token slices
func findCommonPrefix(a, b []int32) int {
	commonLen := 0
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			break
		}
		commonLen++
	}
	return commonLen
}

// Helper function for regular generation
func (m *Model) generateWithConfig(prompt string, config generateConfig, callback func(string) bool) (string, error) {
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

	// Convert prompt to C string
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

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

	params := C.llama_wrapper_generate_params{
		prompt:                cPrompt,
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

	// Call simple generation (handles tokenisation and prefix caching in C++)
	result := C.llama_wrapper_generate(ctx.ptr, params)

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

// Helper function for speculative generation
func (m *Model) generateWithDraftAndConfig(prompt string, draft *Model, config generateConfig, callback func(string) bool) (string, error) {
	// Read lock both models to check closed state
	m.mu.RLock()
	defer m.mu.RUnlock()
	draft.mu.RLock()
	defer draft.mu.RUnlock()

	if m.closed {
		return "", fmt.Errorf("model is closed")
	}
	if draft.closed {
		return "", fmt.Errorf("draft model is closed")
	}

	// Acquire contexts from both pools
	ctx, err := m.pool.acquire()
	if err != nil {
		return "", fmt.Errorf("failed to acquire target context: %w", err)
	}
	defer m.pool.release(ctx)

	draftCtx, err := draft.pool.acquire()
	if err != nil {
		return "", fmt.Errorf("failed to acquire draft context: %w", err)
	}
	defer draft.pool.release(draftCtx)

	// Convert prompt to C string
	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

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

	params := C.llama_wrapper_generate_params{
		prompt:                cPrompt,
		max_tokens:            C.int(config.maxTokens),
		temperature:           C.float(config.temperature),
		top_p:                 C.float(config.topP),
		top_k:                 C.int(config.topK),
		seed:                  C.int(config.seed),
		stop_words:            cStopWords,
		stop_words_count:      stopWordsCount,
		n_draft:               C.int(config.draftTokens),
		debug:                 C.bool(config.debug),
		callback_handle:       callbackHandle,
		enable_prefix_caching: C.bool(config.prefixCaching),
	}

	// Call simple speculative generation (handles tokenisation and prefix caching in C++)
	result := C.llama_wrapper_generate_draft(ctx.ptr, draftCtx.ptr, params)

	// Clean up handle if it was created
	if callback != nil {
		handle.Delete()
	}

	if result == nil {
		return "", fmt.Errorf("speculative generation failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	defer C.llama_wrapper_free_result(result)

	return C.GoString(result), nil
}
