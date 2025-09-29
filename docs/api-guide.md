# API usage guide

This guide covers the llama-go API in detail, showing how to integrate LLM inference into Go
applications.

## Basic usage

The API uses functional options for clean, readable configuration:

```go
package main

import (
    "fmt"
    llama "github.com/tcpipuk/llama-go"
)

func main() {
    // Load model with options
    model, err := llama.LoadModel(
        "/path/to/model.gguf",
        llama.WithF16Memory(),
        llama.WithContext(512),
        llama.WithGPULayers(32),
    )
    if err != nil {
        panic(err)
    }
    defer model.Close()

    // Generate text
    response, err := model.Generate("Hello world", llama.WithMaxTokens(50))
    if err != nil {
        panic(err)
    }

    fmt.Println(response)
}
```

When building your application, set these environment variables:

```bash
export LIBRARY_PATH=$PWD
export C_INCLUDE_PATH=$PWD
export LD_LIBRARY_PATH=$PWD
```

## Model configuration options

### Memory settings

```go
model, err := llama.LoadModel(
    "/path/to/model.gguf",
    llama.WithF16Memory(),        // Use 16-bit floats to reduce memory usage
    llama.WithEmbeddings(),       // Enable embeddings calculation
    llama.WithMMap(true),         // Use memory mapping (default: true)
    llama.WithMLock(),            // Lock memory to prevent swapping
)
```

### Context and batch sizes

```go
model, err := llama.LoadModel(
    "/path/to/model.gguf",
    llama.WithContext(2048),      // Set context window size
    llama.WithBatch(512),         // Set batch processing size
    llama.WithThreads(8),         // Number of threads for processing
)
```

### GPU acceleration

```go
model, err := llama.LoadModel(
    "/path/to/model.gguf",
    llama.WithGPULayers(32),      // Number of layers to run on GPU
    llama.WithMainGPU("0"),       // Which GPU to use (multi-GPU systems)
    llama.WithTensorSplit("0.7,0.3"), // Split tensors across GPUs
)
```

For GPU acceleration setup and build instructions, see the [building guide](building.md).

## Text generation options

### Basic generation

```go
response, err := model.Generate(
    "The capital of France is",
    llama.WithMaxTokens(50),      // Maximum tokens to generate
)
```

### Sampling parameters

```go
response, err := model.Generate(
    "Write a story about:",
    llama.WithMaxTokens(200),
    llama.WithTemperature(0.8),   // Randomness (0.0 = deterministic, 2.0 = very random)
    llama.WithTopK(40),           // Top-K sampling
    llama.WithTopP(0.95),         // Top-P (nucleus) sampling
    llama.WithSeed(42),           // Seed for reproducible output
    llama.WithDebug(),            // Enable debug output
)
```

### Stop sequences

```go
response, err := model.Generate(
    "Q: What is AI?\nA:",
    llama.WithMaxTokens(100),
    llama.WithStopWords("Q:", "\n\n"),  // Stop generation at these sequences
)
```

## Streaming generation

Generate text with real-time token streaming:

```go
err := model.GenerateStream(
    "Tell me a joke:",
    func(token string) bool {
        fmt.Print(token)          // Print tokens as they're generated
        return true               // Return false to stop generation early
    },
    llama.WithMaxTokens(100),
)
```

## Speculative sampling

Speculative sampling uses a small, fast "draft" model to predict multiple tokens ahead, which the
larger "target" model then verifies in parallel. This can speed up generation by 2-3x with the same
quality output, since verifying multiple tokens is faster than generating them one-by-one.

Use this when you want better performance from a large model but don't want to sacrifice quality:

```go
// Load target model (the one you actually want output from)
model, err := llama.LoadModel("/path/to/large-model.gguf")
if err != nil {
    panic(err)
}
defer model.Close()

// Load smaller draft model (used to predict ahead)
draft, err := llama.LoadModel("/path/to/small-model.gguf")
if err != nil {
    panic(err)
}
defer draft.Close()

// Generate with speculative sampling
response, err := model.GenerateWithDraft(
    "Write a story:",
    draft,
    llama.WithMaxTokens(200),
    llama.WithDraftTokens(16),    // Number of draft tokens to predict ahead
)
```

The draft model should be significantly smaller but from the same model family. For example, try
[Gemma 3 1B](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf) as the draft with
[Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-gguf) as the target.

See [examples/speculative](../examples/speculative/) for a complete working implementation.

## Working with embeddings

Generate embeddings for text:

```go
model, err := llama.LoadModel(
    "/path/to/embedding-model.gguf",
    llama.WithEmbeddings(),       // Required for embeddings
)
if err != nil {
    panic(err)
}
defer model.Close()

embeddings, err := model.GetEmbeddings("Hello world")
if err != nil {
    panic(err)
}

fmt.Printf("Embeddings: %v\n", embeddings)
```

See [examples/embedding](../examples/embedding/) for a complete working implementation.

## Advanced usage patterns

### Conversation handling

```go
type Conversation struct {
    model   *llama.Model
    history strings.Builder
}

func (c *Conversation) Ask(question string) (string, error) {
    // Build conversation context
    prompt := c.history.String() + "Human: " + question + "\nAssistant: "

    response, err := c.model.Generate(
        prompt,
        llama.WithMaxTokens(200),
        llama.WithStopWords("Human:", "\n\n"),
    )
    if err != nil {
        return "", err
    }

    // Update conversation history
    c.history.WriteString("Human: " + question + "\n")
    c.history.WriteString("Assistant: " + response + "\n")

    return response, nil
}
```

### Batch processing

For processing multiple prompts efficiently:

```go
prompts := []string{
    "Summarise: The quick brown fox...",
    "Translate to French: Hello world",
    "Complete: The weather today is...",
}

for i, prompt := range prompts {
    response, err := model.Generate(
        prompt,
        llama.WithMaxTokens(100),
        llama.WithSeed(i),  // Different seed for each prompt
    )
    if err != nil {
        fmt.Printf("Error with prompt %d: %v\n", i, err)
        continue
    }

    fmt.Printf("Prompt %d: %s\n", i, response)
}
```

### Streaming to HTTP or WebSocket

For real-time streaming (useful for web services):

```go
func streamResponse(model *llama.Model, prompt string) error {
    err := model.GenerateStream(
        prompt,
        func(token string) bool {
            // Stream to HTTP response, WebSocket, etc.
            fmt.Print(token)
            return true
        },
        llama.WithMaxTokens(200),
    )
    return err
}
```

See [examples/streaming](../examples/streaming/) for a complete working implementation.

## Error handling

Common error patterns and how to handle them:

```go
model, err := llama.LoadModel("/path/to/model.gguf")
if err != nil {
    if strings.Contains(err.Error(), "not found") {
        log.Fatal("Model file not found - check the path")
    }
    if strings.Contains(err.Error(), "invalid format") {
        log.Fatal("Invalid model format - ensure it's a GGUF file")
    }
    log.Fatalf("Failed to load model: %v", err)
}

response, err := model.Generate("test prompt", llama.WithMaxTokens(50))
if err != nil {
    if strings.Contains(err.Error(), "context") {
        log.Println("Context limit exceeded - reduce prompt length or increase context size")
    }
    if strings.Contains(err.Error(), "memory") {
        log.Println("Out of memory - try reducing batch size or GPU layers")
    }
    return fmt.Errorf("generation failed: %w", err)
}
```

## Performance optimisation

### Memory management

```go
// For long-running applications, close and reload models as needed
func reloadModel(currentModel *llama.Model, modelPath string) (*llama.Model, error) {
    currentModel.Close()  // Free memory before loading new model

    return llama.LoadModel(
        modelPath,
        llama.WithF16Memory(),     // Reduce memory usage
        llama.WithContext(1024),   // Reasonable context size
    )
}
```

### GPU optimisation

```go
// For multi-GPU systems
model, err := llama.LoadModel(
    "/path/to/large-model.gguf",
    llama.WithGPULayers(-1),           // Use all available GPU layers
    llama.WithMainGPU("0"),            // Primary GPU
    llama.WithTensorSplit("0.6,0.4"),  // Split workload across GPUs
)
```

### Threading

```go
// Match thread count to your system
import "runtime"

model, err := llama.LoadModel(
    "/path/to/model.gguf",
    llama.WithThreads(runtime.NumCPU()), // Use all CPU cores
)
```

## Integration with web services

Example HTTP handler:

```go
func handleGenerate(model *llama.Model) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        prompt := r.URL.Query().Get("prompt")
        if prompt == "" {
            http.Error(w, "Missing prompt parameter", http.StatusBadRequest)
            return
        }

        response, err := model.Generate(
            prompt,
            llama.WithMaxTokens(200),
            llama.WithTemperature(0.7),
        )
        if err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }

        w.Header().Set("Content-Type", "text/plain")
        w.Write([]byte(response))
    }
}
```

For streaming responses to HTTP clients or WebSocket connections, use `GenerateStream` with a
callback that writes to your response writer.

## Thread safety

The library is not thread-safe. For concurrent access, use a pool pattern:

```go
type ModelPool struct {
    models chan *llama.Model
}

func NewModelPool(modelPath string, size int) (*ModelPool, error) {
    pool := &ModelPool{
        models: make(chan *llama.Model, size),
    }

    for i := 0; i < size; i++ {
        model, err := llama.LoadModel(modelPath, llama.WithF16Memory())
        if err != nil {
            return nil, err
        }
        pool.models <- model
    }

    return pool, nil
}

func (p *ModelPool) Generate(prompt string, options ...llama.GenerateOption) (string, error) {
    model := <-p.models        // Get model from pool
    defer func() {
        p.models <- model      // Return model to pool
    }()

    return model.Generate(prompt, options...)
}
```

This allows safe concurrent usage across multiple goroutines whilst maintaining good performance.
