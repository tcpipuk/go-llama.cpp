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
	"unsafe"
)

// Model represents a loaded LLAMA model with its context
type Model struct {
	ptr unsafe.Pointer
}

// modelConfig holds configuration for model loading
type modelConfig struct {
	contextSize int
	batchSize   int
	gpuLayers   int
	threads     int
	f16Memory   bool
	mlock       bool
	mmap        bool
	embeddings  bool
	mainGPU     string
	tensorSplit string
}

// generateConfig holds configuration for text generation
type generateConfig struct {
	maxTokens   int
	temperature float32
	topP        float32
	topK        int
	seed        int
	stopWords   []string
	draftTokens int
	debug       bool
}

// Default configurations
var defaultModelConfig = modelConfig{
	contextSize: 2048,
	batchSize:   512,
	gpuLayers:   0,
	threads:     runtime.NumCPU(),
	f16Memory:   false,
	mlock:       false,
	mmap:        true,
	embeddings:  false,
}

var defaultGenerateConfig = generateConfig{
	maxTokens:   128,
	temperature: 0.8,
	topP:        0.95,
	topK:        40,
	seed:        -1,
	draftTokens: 16,
	debug:       false,
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

// Model loading options
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

func WithThreads(n int) ModelOption {
	return func(c *modelConfig) {
		c.threads = n
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

// LoadModel loads a GGUF model from the specified path
func LoadModel(path string, opts ...ModelOption) (*Model, error) {
	config := defaultModelConfig
	for _, opt := range opts {
		opt(&config)
	}

	// Convert Go config to C struct
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
		n_ctx:        C.int(config.contextSize),
		n_batch:      C.int(config.batchSize),
		n_gpu_layers: C.int(config.gpuLayers),
		n_threads:    C.int(config.threads),
		f16_memory:   C.bool(config.f16Memory),
		mlock:        C.bool(config.mlock),
		mmap:         C.bool(config.mmap),
		embeddings:   C.bool(config.embeddings),
		main_gpu:     cMainGPU,
		tensor_split: cTensorSplit,
	}

	ptr := C.llama_wrapper_model_load(cPath, params)
	if ptr == nil {
		return nil, fmt.Errorf("failed to load model: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	model := &Model{ptr: ptr}

	// Set finalizer to ensure cleanup
	runtime.SetFinalizer(model, (*Model).Close)

	return model, nil
}

// Close frees the model and its associated resources
func (m *Model) Close() error {
	if m.ptr != nil {
		C.llama_wrapper_model_free(m.ptr)
		m.ptr = nil
		runtime.SetFinalizer(m, nil)
	}
	return nil
}

// Generate generates text from the given prompt
func (m *Model) Generate(prompt string, opts ...GenerateOption) (string, error) {
	if m.ptr == nil {
		return "", fmt.Errorf("model is closed")
	}

	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return m.generateWithConfig(prompt, config, nil)
}

// GenerateStream generates text with streaming output via callback
func (m *Model) GenerateStream(prompt string, callback func(token string) bool, opts ...GenerateOption) error {
	if m.ptr == nil {
		return fmt.Errorf("model is closed")
	}

	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	_, err := m.generateWithConfig(prompt, config, callback)
	return err
}

// GenerateWithDraft performs speculative generation using a draft model
func (m *Model) GenerateWithDraft(prompt string, draft *Model, opts ...GenerateOption) (string, error) {
	if m.ptr == nil {
		return "", fmt.Errorf("model is closed")
	}
	if draft.ptr == nil {
		return "", fmt.Errorf("draft model is closed")
	}

	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return m.generateWithDraftAndConfig(prompt, draft, config, nil)
}

// GenerateWithDraftStream performs speculative generation with streaming output
func (m *Model) GenerateWithDraftStream(prompt string, draft *Model, callback func(token string) bool, opts ...GenerateOption) error {
	if m.ptr == nil {
		return fmt.Errorf("model is closed")
	}
	if draft.ptr == nil {
		return fmt.Errorf("draft model is closed")
	}

	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	_, err := m.generateWithDraftAndConfig(prompt, draft, config, callback)
	return err
}

// Tokenize converts text to tokens
func (m *Model) Tokenize(text string) ([]int32, error) {
	if m.ptr == nil {
		return nil, fmt.Errorf("model is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	maxTokens := 8192 // Reasonable upper bound
	tokens := make([]C.int, maxTokens)

	count := C.llama_wrapper_tokenize(m.ptr, cText, &tokens[0], C.int(maxTokens))
	if count < 0 {
		return nil, fmt.Errorf("tokenization failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	result := make([]int32, count)
	for i := 0; i < int(count); i++ {
		result[i] = int32(tokens[i])
	}

	return result, nil
}

// GetEmbeddings computes embeddings for the given text
func (m *Model) GetEmbeddings(text string) ([]float32, error) {
	if m.ptr == nil {
		return nil, fmt.Errorf("model is closed")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	maxEmbeddings := 4096 // Reasonable upper bound
	embeddings := make([]C.float, maxEmbeddings)

	count := C.llama_wrapper_embeddings(m.ptr, cText, &embeddings[0], C.int(maxEmbeddings))
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
		prompt:           cPrompt,
		max_tokens:       C.int(config.maxTokens),
		temperature:      C.float(config.temperature),
		top_p:            C.float(config.topP),
		top_k:            C.int(config.topK),
		seed:             C.int(config.seed),
		stop_words:       cStopWords,
		stop_words_count: stopWordsCount,
		debug:            C.bool(config.debug),
		callback_handle:  callbackHandle,
	}

	result := C.llama_wrapper_generate(m.ptr, params)

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
		prompt:           cPrompt,
		max_tokens:       C.int(config.maxTokens),
		temperature:      C.float(config.temperature),
		top_p:            C.float(config.topP),
		top_k:            C.int(config.topK),
		seed:             C.int(config.seed),
		stop_words:       cStopWords,
		stop_words_count: stopWordsCount,
		n_draft:          C.int(config.draftTokens),
		debug:            C.bool(config.debug),
		callback_handle:  callbackHandle,
	}

	result := C.llama_wrapper_generate_draft(m.ptr, draft.ptr, params)

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
