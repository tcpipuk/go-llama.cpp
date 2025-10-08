package llama

import (
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

/*
#cgo CFLAGS: -I./llama.cpp -I./ -I./llama.cpp/ggml/include -I./llama.cpp/include -I./llama.cpp/common
#cgo CPPFLAGS: -I./llama.cpp -I./ -I./llama.cpp/ggml/include -I./llama.cpp/include -I./llama.cpp/common
#cgo LDFLAGS: -L./ -lcommon -lllama -lggml-cpu -lggml-base -lggml -lstdc++ -lm
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

// context represents a single execution context with usage tracking
type context struct {
	ptr     unsafe.Pointer // llama_wrapper_context_t*
	lastUse time.Time
	mu      sync.Mutex
}

// contextPool manages a dynamic pool of contexts for a model
type contextPool struct {
	model     unsafe.Pointer // llama_wrapper_model_t* (weights only)
	config    modelConfig
	contexts  []*context
	available chan *context
	done      chan struct{}
	mu        sync.Mutex
	wg        sync.WaitGroup
}

// Model represents a loaded LLAMA model with concurrent inference support.
//
// Model instances are thread-safe for concurrent inference operations but NOT
// for concurrent Close() calls. The internal context pool manages multiple
// execution contexts, allowing parallel generations when configured with
// WithPoolSize.
//
// Resources are automatically freed via finaliser, but explicit Close() is
// recommended for deterministic cleanup:
//
//	model, _ := llama.LoadModel("model.gguf")
//	defer model.Close()
//
// Note: Calling methods after Close() returns an error.
type Model struct {
	modelPtr unsafe.Pointer // llama_wrapper_model_t* (weights only)
	pool     *contextPool
	mu       sync.RWMutex
	closed   bool
}

// modelConfig holds configuration for model loading
type modelConfig struct {
	contextSize   int
	batchSize     int
	gpuLayers     int
	threads       int
	threadsBatch  int
	f16Memory     bool
	mlock         bool
	mmap          bool
	embeddings    bool
	mainGPU       string
	tensorSplit   string
	minContexts   int
	maxContexts   int
	idleTimeout   time.Duration
	prefixCaching bool // Enable KV cache prefix reuse (default: true)
}

// generateConfig holds configuration for text generation
type generateConfig struct {
	maxTokens     int
	temperature   float32
	topP          float32
	topK          int
	seed          int
	stopWords     []string
	draftTokens   int
	debug         bool
	prefixCaching bool // Per-generation prefix caching control
}

// Default configurations
var defaultModelConfig = modelConfig{
	contextSize:   0,    // 0 = use model's native maximum (queried after load)
	batchSize:     512,
	gpuLayers:     -1, // Offload all layers to GPU by default (falls back to CPU if unavailable)
	threads:       runtime.NumCPU(),
	threadsBatch:  0, // 0 means use same as threads (set in wrapper)
	f16Memory:     false,
	mlock:         false,
	mmap:          true,
	embeddings:    false,
	minContexts:   1,
	maxContexts:   1,
	idleTimeout:   1 * time.Minute,
	prefixCaching: true, // Enable by default for performance
}

var defaultGenerateConfig = generateConfig{
	maxTokens:     128,
	temperature:   0.8,
	topP:          0.95,
	topK:          40,
	seed:          -1,
	draftTokens:   16,
	debug:         false,
	prefixCaching: true, // Inherit from model default
}

// ModelOption configures model loading behaviour.
//
// Options are applied using the functional options pattern. Available options
// include WithContext, WithGPULayers, WithThreads, WithPoolSize, and others.
// See individual With* functions for details.
//
// Example:
//
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithContext(8192),
//	    llama.WithGPULayers(35),
//	    llama.WithThreads(8),
//	)
type ModelOption func(*modelConfig)

// GenerateOption configures text generation behaviour.
//
// Options are applied using the functional options pattern. Available options
// include WithMaxTokens, WithTemperature, WithTopP, WithTopK, WithSeed,
// WithStopWords, WithDraftTokens, WithDebug, and WithPrefixCaching.
// See individual With* functions for details.
//
// Example:
//
//	text, err := model.Generate("Write a story",
//	    llama.WithMaxTokens(256),
//	    llama.WithTemperature(0.7),
//	)
type GenerateOption func(*generateConfig)

// LoadModel loads a GGUF model from the specified path.
//
// The path must point to a valid GGUF format model file. Legacy GGML formats
// are not supported. The function applies the provided options using the
// functional options pattern, with sensible defaults if none are specified.
//
// Resources are managed automatically via finaliser, but explicit cleanup with
// Close() is recommended for deterministic resource management:
//
//	model, err := llama.LoadModel("model.gguf")
//	if err != nil {
//	    return err
//	}
//	defer model.Close()
//
// Returns an error if the file doesn't exist, is not a valid GGUF model, or
// if context pool initialisation fails.
//
// Examples:
//
//	// Load with defaults
//	model, err := llama.LoadModel("model.gguf")
//
//	// Load with custom configuration
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithContext(8192),
//	    llama.WithGPULayers(35),
//	    llama.WithPoolSize(1, 4),
//	)
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	if path == "" {
		return nil, fmt.Errorf("Model path cannot be null")
	}

	config := defaultModelConfig
	for _, opt := range opts {
		opt(&config)
	}

	// Convert Go config to C struct for model loading
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var cMainGPU *C.char
	if config.mainGPU != "" {
		cMainGPU = C.CString(config.mainGPU)
		defer C.free(unsafe.Pointer(cMainGPU))
	}

	var cTensorSplit *C.char
	if config.tensorSplit != "" {
		cTensorSplit = C.CString(config.tensorSplit)
		defer C.free(unsafe.Pointer(cTensorSplit))
	}

	params := C.llama_wrapper_model_params{
		n_ctx:          C.int(config.contextSize),
		n_batch:        C.int(config.batchSize),
		n_gpu_layers:   C.int(config.gpuLayers),
		n_threads:      C.int(config.threads),
		n_threads_batch: C.int(config.threadsBatch),
		f16_memory:     C.bool(config.f16Memory),
		mlock:          C.bool(config.mlock),
		mmap:           C.bool(config.mmap),
		embeddings:     C.bool(config.embeddings),
		main_gpu:       cMainGPU,
		tensor_split:   cTensorSplit,
	}

	// Load model (weights only)
	modelPtr := C.llama_wrapper_model_load(cPath, params)
	if modelPtr == nil {
		return nil, fmt.Errorf("failed to load model: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	// Query model's native context if user didn't specify
	if config.contextSize == 0 {
		nativeContext := int(C.llama_wrapper_get_model_context_length(modelPtr))
		config.contextSize = nativeContext
	}

	// Optimisation: clamp batch size to context size
	// You can never process more tokens per batch than fit in total context
	if config.batchSize > config.contextSize {
		config.batchSize = config.contextSize
	}

	// Create context pool
	pool, err := newContextPool(modelPtr, config)
	if err != nil {
		C.llama_wrapper_model_free(modelPtr)
		return nil, fmt.Errorf("failed to create context pool: %w", err)
	}

	model := &Model{
		modelPtr: modelPtr,
		pool:     pool,
	}

	// Set finaliser to ensure cleanup
	runtime.SetFinalizer(model, (*Model).Close)

	return model, nil
}

// Close frees the model and its associated resources.
//
// This method is idempotent - multiple calls are safe and subsequent calls
// return immediately without error. Close() blocks until all active generation
// requests complete, ensuring clean shutdown.
//
// After Close() is called, all other methods return an error. The method uses
// a write lock to prevent concurrent operations during cleanup.
//
// Example:
//
//	model, _ := llama.LoadModel("model.gguf")
//	defer model.Close()
func (m *Model) Close() error {
	m.mu.Lock() // Write lock to block all operations
	defer m.mu.Unlock()

	if m.closed {
		return nil
	}

	// Remove finaliser FIRST to prevent race with GC
	runtime.SetFinalizer(m, nil)

	// Close pool (frees all contexts)
	if m.pool != nil {
		m.pool.close()
		m.pool = nil
	}

	// Free model
	if m.modelPtr != nil {
		C.llama_wrapper_model_free(m.modelPtr)
		m.modelPtr = nil
	}

	m.closed = true
	return nil
}
