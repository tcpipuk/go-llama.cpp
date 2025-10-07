package llama

/*
#cgo CFLAGS: -I./llama.cpp -I./ -I./llama.cpp/ggml/include -I./llama.cpp/include -I./llama.cpp/common
#cgo CPPFLAGS: -I./llama.cpp -I./ -I./llama.cpp/ggml/include -I./llama.cpp/include -I./llama.cpp/common
#cgo LDFLAGS: -L./ -lcommon -lllama -lggml-cpu -lggml-base -lggml -lstdc++ -lm
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"runtime/cgo"
	"sync"
	"time"
	"unsafe"
)

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

// Model represents a loaded LLAMA model with its context pool
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

// ModelOption configures model loading
type ModelOption func(*modelConfig)

// GenerateOption configures text generation
type GenerateOption func(*generateConfig)

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

// newContextPool creates a new context pool with min contexts initialised
func newContextPool(model unsafe.Pointer, config modelConfig) (*contextPool, error) {
	pool := &contextPool{
		model:     model,
		config:    config,
		contexts:  make([]*context, 0, config.maxContexts),
		available: make(chan *context, config.maxContexts),
		done:      make(chan struct{}),
	}

	// Convert Go config to C struct for context creation
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

	// Create min contexts upfront
	for i := 0; i < config.minContexts; i++ {
		ctxPtr := C.llama_wrapper_context_create(model, params)
		if ctxPtr == nil {
			// Clean up already created contexts
			pool.close()
			return nil, fmt.Errorf("failed to create initial context %d: %s", i, C.GoString(C.llama_wrapper_last_error()))
		}

		ctx := &context{
			ptr:     ctxPtr,
			lastUse: time.Now(),
		}
		pool.contexts = append(pool.contexts, ctx)
		pool.available <- ctx
	}

	// Start cleanup goroutine
	pool.wg.Add(1)
	go pool.cleanup()

	return pool, nil
}

// acquire gets a context from the pool, creating one if needed
func (p *contextPool) acquire() (*context, error) {
	select {
	case ctx := <-p.available:
		// Got an available context
		ctx.mu.Lock()
		return ctx, nil
	case <-p.done:
		return nil, fmt.Errorf("pool is closed")
	default:
		// No available context - try to create one
		p.mu.Lock()
		if len(p.contexts) < p.config.maxContexts {
			// We can create a new context
			var cMainGPU *C.char
			if p.config.mainGPU != "" {
				cMainGPU = C.CString(p.config.mainGPU)
				defer C.free(unsafe.Pointer(cMainGPU))
			}

			var cTensorSplit *C.char
			if p.config.tensorSplit != "" {
				cTensorSplit = C.CString(p.config.tensorSplit)
				defer C.free(unsafe.Pointer(cTensorSplit))
			}

			params := C.llama_wrapper_model_params{
				n_ctx:          C.int(p.config.contextSize),
				n_batch:        C.int(p.config.batchSize),
				n_gpu_layers:   C.int(p.config.gpuLayers),
				n_threads:      C.int(p.config.threads),
				n_threads_batch: C.int(p.config.threadsBatch),
				f16_memory:     C.bool(p.config.f16Memory),
				mlock:          C.bool(p.config.mlock),
				mmap:           C.bool(p.config.mmap),
				embeddings:     C.bool(p.config.embeddings),
				main_gpu:       cMainGPU,
				tensor_split:   cTensorSplit,
			}

			ctxPtr := C.llama_wrapper_context_create(p.model, params)
			if ctxPtr == nil {
				p.mu.Unlock()
				return nil, fmt.Errorf("failed to create context: %s", C.GoString(C.llama_wrapper_last_error()))
			}

			ctx := &context{
				ptr:     ctxPtr,
				lastUse: time.Now(),
			}
			p.contexts = append(p.contexts, ctx)
			p.mu.Unlock()

			ctx.mu.Lock()
			return ctx, nil
		}
		p.mu.Unlock()

		// Max contexts reached - wait for one to become available
		select {
		case ctx := <-p.available:
			ctx.mu.Lock()
			return ctx, nil
		case <-p.done:
			return nil, fmt.Errorf("pool is closed")
		}
	}
}

// release returns a context to the pool
func (p *contextPool) release(ctx *context) {
	ctx.lastUse = time.Now()
	ctx.mu.Unlock()

	select {
	case p.available <- ctx:
		// Context returned to pool
	case <-p.done:
		// Pool is closed, don't return
	}
}

// cleanup runs periodically to destroy idle contexts
func (p *contextPool) cleanup() {
	defer p.wg.Done()

	// Wait for shutdown signal
	// Note: Idle context cleanup is disabled to avoid complexity during testing
	// In production, this could be re-enabled with proper synchronisation
	<-p.done
}

// close shuts down the pool and frees all contexts
func (p *contextPool) close() {
	close(p.done)
	p.wg.Wait()

	p.mu.Lock()
	defer p.mu.Unlock()

	// Drain the available channel before closing
	for len(p.available) > 0 {
		<-p.available
	}
	close(p.available)

	// Free all contexts
	for _, ctx := range p.contexts {
		if ctx.ptr != nil {
			C.llama_wrapper_context_free(ctx.ptr)
			ctx.ptr = nil // Prevent double-free
		}
	}
	p.contexts = nil
}

// Model loading options

// WithContext sets the context window size in tokens.
//
// The context size determines how many tokens (prompt + generation) the model
// can process. By default, the library uses the model's native maximum context
// length (e.g. 32768 for Qwen3, 128000 for Gemma 3 models >4B).
//
// Override this if you need to limit memory usage or have specific requirements.
//
// IMPORTANT: Very small context sizes (< 64 tokens) may cause llama.cpp to
// crash internally. The library provides defensive checks but cannot prevent
// all edge cases with absurdly small contexts.
//
// Default: 0 (uses model's native maximum from GGUF metadata)
//
// Examples:
//
//	// Use model's full capability (default)
//	model, err := llama.LoadModel("model.gguf")
//
//	// Limit to 8K for memory savings
//	model, err := llama.LoadModel("model.gguf",
//	    llama.WithContext(8192),
//	)
func WithContext(size int) ModelOption {
	return func(c *modelConfig) {
		c.contextSize = size
	}
}

func WithBatch(size int) ModelOption {
	return func(c *modelConfig) {
		c.batchSize = size
	}
}

func WithGPULayers(n int) ModelOption {
	return func(c *modelConfig) {
		c.gpuLayers = n
	}
}

// WithThreads sets the number of threads for token generation.
// If not specified, defaults to runtime.NumCPU().
// This also sets threadsBatch to the same value unless WithThreadsBatch is used.
func WithThreads(n int) ModelOption {
	return func(c *modelConfig) {
		c.threads = n
	}
}

// WithThreadsBatch sets the number of threads for batch/prompt processing.
// If not specified, defaults to the same value as threads.
// For most use cases, leaving this unset is recommended.
func WithThreadsBatch(n int) ModelOption {
	return func(c *modelConfig) {
		c.threadsBatch = n
	}
}

func WithF16Memory() ModelOption {
	return func(c *modelConfig) {
		c.f16Memory = true
	}
}

func WithMLock() ModelOption {
	return func(c *modelConfig) {
		c.mlock = true
	}
}

func WithMMap(enabled bool) ModelOption {
	return func(c *modelConfig) {
		c.mmap = enabled
	}
}

func WithEmbeddings() ModelOption {
	return func(c *modelConfig) {
		c.embeddings = true
	}
}

func WithMainGPU(gpu string) ModelOption {
	return func(c *modelConfig) {
		c.mainGPU = gpu
	}
}

func WithTensorSplit(split string) ModelOption {
	return func(c *modelConfig) {
		c.tensorSplit = split
	}
}

func WithPoolSize(min, max int) ModelOption {
	return func(c *modelConfig) {
		if min < 1 {
			min = 1
		}
		if max < min {
			max = min
		}
		c.minContexts = min
		c.maxContexts = max
	}
}

func WithIdleTimeout(d time.Duration) ModelOption {
	return func(c *modelConfig) {
		c.idleTimeout = d
	}
}

// Generation options
func WithMaxTokens(n int) GenerateOption {
	return func(c *generateConfig) {
		c.maxTokens = n
	}
}

func WithTemperature(t float32) GenerateOption {
	return func(c *generateConfig) {
		c.temperature = t
	}
}

func WithTopP(p float32) GenerateOption {
	return func(c *generateConfig) {
		c.topP = p
	}
}

func WithTopK(k int) GenerateOption {
	return func(c *generateConfig) {
		c.topK = k
	}
}

func WithSeed(seed int) GenerateOption {
	return func(c *generateConfig) {
		c.seed = seed
	}
}

func WithStopWords(words ...string) GenerateOption {
	return func(c *generateConfig) {
		c.stopWords = words
	}
}

func WithDraftTokens(n int) GenerateOption {
	return func(c *generateConfig) {
		c.draftTokens = n
	}
}

func WithDebug() GenerateOption {
	return func(c *generateConfig) {
		c.debug = true
	}
}

// WithPrefixCaching enables or disables KV cache prefix reuse.
// When enabled (default), the library reuses KV cache entries for matching
// prompt prefixes, significantly improving performance for conversation-style
// usage or repeated prompts.
//
// Disable this if you need guaranteed-fresh state for every generation,
// though the performance cost is minimal with the last-token refresh optimisation.
//
// Default: true (enabled)
func WithPrefixCaching(enabled bool) GenerateOption {
	return func(c *generateConfig) {
		c.prefixCaching = enabled
	}
}

// LoadModel loads a GGUF model from the specified path
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

// Close frees the model and its associated resources
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

// Generate generates text from the given prompt
func (m *Model) Generate(prompt string, opts ...GenerateOption) (string, error) {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return m.generateWithConfig(prompt, config, nil)
}

// GenerateStream generates text with streaming output via callback
func (m *Model) GenerateStream(prompt string, callback func(token string) bool, opts ...GenerateOption) error {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	_, err := m.generateWithConfig(prompt, config, callback)
	return err
}

// GenerateWithDraft performs speculative generation using a draft model
func (m *Model) GenerateWithDraft(prompt string, draft *Model, opts ...GenerateOption) (string, error) {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return m.generateWithDraftAndConfig(prompt, draft, config, nil)
}

// GenerateWithDraftStream performs speculative generation with streaming output
func (m *Model) GenerateWithDraftStream(prompt string, draft *Model, callback func(token string) bool, opts ...GenerateOption) error {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	_, err := m.generateWithDraftAndConfig(prompt, draft, config, callback)
	return err
}

// GenerateWithTokens generates text from pre-tokenised input (advanced API)
// This allows callers to manage tokenisation themselves and provides manual control
// over prefix caching. For most use cases, use Generate() instead.
func (m *Model) GenerateWithTokens(tokens []int32, opts ...GenerateOption) (string, error) {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return m.generateWithTokensAndConfig(tokens, config, nil)
}

// GenerateWithTokensStream generates text with streaming output via callback (advanced API)
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

// Tokenize converts text to tokens
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
		return nil, fmt.Errorf("tokenization failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	result := make([]int32, count)
	for i := 0; i < int(count); i++ {
		result[i] = int32(tokens[i])
	}

	return result, nil
}

// GetCachedTokenCount returns the number of cached tokens in a context (for debugging/metrics)
// Note: This requires acquiring a context from the pool, so the count may vary between calls
// depending on which context is available.
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

// GetEmbeddings computes embeddings for the given text
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
