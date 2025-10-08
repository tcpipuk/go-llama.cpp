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

	// Convert DRY sequence breakers to C array
	var cDryBreakers **C.char
	var dryBreakersCount C.int
	if len(config.drySequenceBreakers) > 0 {
		dryBreakersCount = C.int(len(config.drySequenceBreakers))
		cDryBreakersArray := make([]*C.char, len(config.drySequenceBreakers))
		for i, breaker := range config.drySequenceBreakers {
			cDryBreakersArray[i] = C.CString(breaker)
		}
		defer func() {
			for _, ptr := range cDryBreakersArray {
				C.free(unsafe.Pointer(ptr))
			}
		}()
		cDryBreakers = (**C.char)(unsafe.Pointer(&cDryBreakersArray[0]))
	}

	params := C.llama_wrapper_generate_params{
		prompt:                cPrompt,
		max_tokens:            C.int(config.maxTokens),
		seed:                  C.int(config.seed),
		stop_words:            cStopWords,
		stop_words_count:      stopWordsCount,
		debug:                 C.bool(config.debug),
		callback_handle:       callbackHandle,
		enable_prefix_caching: C.bool(config.prefixCaching),

		// Basic sampling parameters
		temperature: C.float(config.temperature),
		top_k:       C.int(config.topK),
		top_p:       C.float(config.topP),
		min_p:       C.float(config.minP),
		typ_p:       C.float(config.typP),
		top_n_sigma: C.float(config.topNSigma),
		min_keep:    C.int(config.minKeep),

		// Repetition penalties
		penalty_last_n:  C.int(config.penaltyLastN),
		penalty_repeat:  C.float(config.penaltyRepeat),
		penalty_freq:    C.float(config.penaltyFreq),
		penalty_present: C.float(config.penaltyPresent),

		// DRY sampling
		dry_multiplier:              C.float(config.dryMultiplier),
		dry_base:                    C.float(config.dryBase),
		dry_allowed_length:          C.int(config.dryAllowedLength),
		dry_penalty_last_n:          C.int(config.dryPenaltyLastN),
		dry_sequence_breakers:       cDryBreakers,
		dry_sequence_breakers_count: dryBreakersCount,

		// Dynamic temperature
		dynatemp_range:    C.float(config.dynatempRange),
		dynatemp_exponent: C.float(config.dynatempExponent),

		// XTC sampling
		xtc_probability: C.float(config.xtcProbability),
		xtc_threshold:   C.float(config.xtcThreshold),

		// Mirostat sampling
		mirostat:     C.int(config.mirostat),
		mirostat_tau: C.float(config.mirostatTau),
		mirostat_eta: C.float(config.mirostatEta),

		// Other parameters
		n_prev:     C.int(config.nPrev),
		n_probs:    C.int(config.nProbs),
		ignore_eos: C.bool(config.ignoreEOS),
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

	// Convert DRY sequence breakers to C array
	var cDryBreakers2 **C.char
	var dryBreakersCount2 C.int
	if len(config.drySequenceBreakers) > 0 {
		dryBreakersCount2 = C.int(len(config.drySequenceBreakers))
		cDryBreakersArray2 := make([]*C.char, len(config.drySequenceBreakers))
		for i, breaker := range config.drySequenceBreakers {
			cDryBreakersArray2[i] = C.CString(breaker)
		}
		defer func() {
			for _, ptr := range cDryBreakersArray2 {
				C.free(unsafe.Pointer(ptr))
			}
		}()
		cDryBreakers2 = (**C.char)(unsafe.Pointer(&cDryBreakersArray2[0]))
	}

	params := C.llama_wrapper_generate_params{
		prompt:                cPrompt,
		max_tokens:            C.int(config.maxTokens),
		seed:                  C.int(config.seed),
		stop_words:            cStopWords,
		stop_words_count:      stopWordsCount,
		n_draft:               C.int(config.draftTokens),
		debug:                 C.bool(config.debug),
		callback_handle:       callbackHandle,
		enable_prefix_caching: C.bool(config.prefixCaching),

		// Basic sampling parameters
		temperature: C.float(config.temperature),
		top_k:       C.int(config.topK),
		top_p:       C.float(config.topP),
		min_p:       C.float(config.minP),
		typ_p:       C.float(config.typP),
		top_n_sigma: C.float(config.topNSigma),
		min_keep:    C.int(config.minKeep),

		// Repetition penalties
		penalty_last_n:  C.int(config.penaltyLastN),
		penalty_repeat:  C.float(config.penaltyRepeat),
		penalty_freq:    C.float(config.penaltyFreq),
		penalty_present: C.float(config.penaltyPresent),

		// DRY sampling
		dry_multiplier:              C.float(config.dryMultiplier),
		dry_base:                    C.float(config.dryBase),
		dry_allowed_length:          C.int(config.dryAllowedLength),
		dry_penalty_last_n:          C.int(config.dryPenaltyLastN),
		dry_sequence_breakers:       cDryBreakers2,
		dry_sequence_breakers_count: dryBreakersCount2,

		// Dynamic temperature
		dynatemp_range:    C.float(config.dynatempRange),
		dynatemp_exponent: C.float(config.dynatempExponent),

		// XTC sampling
		xtc_probability: C.float(config.xtcProbability),
		xtc_threshold:   C.float(config.xtcThreshold),

		// Mirostat sampling
		mirostat:     C.int(config.mirostat),
		mirostat_tau: C.float(config.mirostatTau),
		mirostat_eta: C.float(config.mirostatEta),

		// Other parameters
		n_prev:     C.int(config.nPrev),
		n_probs:    C.int(config.nProbs),
		ignore_eos: C.bool(config.ignoreEOS),
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
