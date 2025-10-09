# llama-go: Run LLMs locally with Go

[![Go Reference](https://pkg.go.dev/badge/github.com/tcpipuk/llama-go.svg)](https://pkg.go.dev/github.com/tcpipuk/llama-go)

Go bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), enabling you to run large
language models locally with GPU acceleration. Production-ready library with thread-safe concurrent
inference and comprehensive test coverage. Integrate LLM inference directly into Go applications
with a clean, idiomatic API.

This is an active fork of [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp),
which hasn't been maintained since October 2023. The goal is keeping Go developers up-to-date with
llama.cpp whilst offering a lighter, more performant alternative to Python-based ML stacks like
PyTorch and/or vLLM.

**Documentation**:

- **Getting started**: [Installation guide](docs/getting-started.md) |
  [Build options](docs/building.md)
- **API reference**: [pkg.go.dev](https://pkg.go.dev/github.com/tcpipuk/llama-go) (complete godoc
  with examples)
- **Examples**: [Working code examples](examples/) for chat, streaming, embeddings, speculative
  decoding
- **Upstream**: [llama.cpp](https://github.com/ggml-org/llama.cpp) for model formats and engine
  details

## Quick start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/tcpipuk/llama-go
cd llama-go

# Build the library
make libbinding.a

# Download a test model
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# Run an example
export LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD LD_LIBRARY_PATH=$PWD
go run ./examples/simple -m Qwen3-0.6B-Q8_0.gguf -p "Hello world" -n 50
```

## Basic usage

```go
package main

import (
    "context"
    "fmt"
    llama "github.com/tcpipuk/llama-go"
)

func main() {
    model, err := llama.LoadModel(
        "/path/to/model.gguf",
        llama.WithF16Memory(),
        llama.WithContext(512),
    )
    if err != nil {
        panic(err)
    }
    defer model.Close()

    // Chat completion (uses model's chat template)
    messages := []llama.ChatMessage{
        {Role: "system", Content: "You are a helpful assistant."},
        {Role: "user", Content: "What is the capital of France?"},
    }
    response, err := model.Chat(context.Background(), messages, llama.ChatOptions{
        MaxTokens: llama.Int(100),
    })
    if err != nil {
        panic(err)
    }
    fmt.Println(response.Content)

    // Or raw text generation
    text, err := model.Generate("Hello world", llama.WithMaxTokens(50))
    if err != nil {
        panic(err)
    }
    fmt.Println(text)
}
```

When building, set these environment variables:

```bash
export LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD LD_LIBRARY_PATH=$PWD
```

## Key capabilities

**Text generation and chat**: Generate text with LLMs using native chat completion (with automatic
chat template formatting) or raw text generation. Extract embeddings for semantic search,
clustering, and similarity tasks.

**GPU acceleration**: Supports NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal), Intel (SYCL), and
cross-platform acceleration (Vulkan, OpenCL). Eight backend options cover virtually all modern GPU
hardware, plus distributed inference via RPC.

**Production ready**: Comprehensive test suite with almost 400 test cases and CI validation
including CUDA builds. Active development tracking llama.cpp releases - maintained for production
use, not a demo project.

**Advanced features**: Cache common prompt prefixes to avoid recomputing system prompts across
thousands of generations. Serve multiple concurrent requests with a single model loaded in VRAM (no
weight duplication). Stream tokens via callbacks or buffered channels (decouples GPU inference from
slow processing). Speculative decoding for 2-3× generation speedup.

## Architecture

The library bridges Go and C++ using CGO, keeping the heavy computation in llama.cpp's optimised
C++ code whilst providing a clean Go API. This minimises CGO overhead whilst maximising
performance.

Key components:

- `wrapper.cpp`/`wrapper.h` - CGO interface to llama.cpp
- Clean Go API with comprehensive godoc comments
- `llama.cpp/` - Git submodule tracking upstream releases

The design uses functional options for configuration, dynamic context pooling for thread safety,
automatic KV cache prefix reuse for performance, resource management with finalizers, and streaming
callbacks via cgo.Handle for safe Go-C interaction.

## Licence

MIT
