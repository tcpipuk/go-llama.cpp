# go-llama.cpp

[![Go Reference](https://pkg.go.dev/badge/github.com/tcpipuk/go-llama.cpp.svg)](https://pkg.go.dev/github.com/tcpipuk/go-llama.cpp)

Go bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), enabling you to run large
language models locally with hardware acceleration support. Integrate LLM inference directly into Go
applications with a clean, idiomatic API.

This is an active fork of [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp),
which hasn't been maintained since October 2023. The goal is keeping Go developers up-to-date with
llama.cpp whilst offering a lighter, more performant alternative to Python-based ML stacks like
PyTorch and/or vLLM.

**Note**: Historical tags use the original module path `github.com/go-skynet/go-llama.cpp`. For
new development, use `github.com/tcpipuk/go-llama.cpp`.

**Releases**: This fork's tags follow llama.cpp releases using the format `llama.cpp-{tag}` (e.g.
`llama.cpp-b6603`). This ensures compatibility tracking with the underlying C++ library.

## Quick start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/tcpipuk/go-llama.cpp
cd go-llama.cpp

# Build the library
make libbinding.a

# Download a test model
wget https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# Run an example
export LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD LD_LIBRARY_PATH=$PWD
go run ./examples/simple -m Qwen3-0.6B-Q8_0.gguf -p "Hello world" -n 50
```

See [getting started guide](docs/getting-started.md) for detailed instructions and
[examples](examples/README.md) for more usage patterns.

## Basic usage

```go
package main

import (
    "fmt"
    llama "github.com/tcpipuk/go-llama.cpp"
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

    response, err := model.Generate("Hello world", llama.WithMaxTokens(50))
    if err != nil {
        panic(err)
    }

    fmt.Println(response)
}
```

When building, set these environment variables:

```bash
export LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD LD_LIBRARY_PATH=$PWD
```

## Thread safety

The library is not thread-safe. For concurrent usage, implement a pool pattern as shown in the [API guide](docs/api-guide.md#thread-safety).

## Documentation

### Essential guides

- **[Getting started](docs/getting-started.md)** - Complete walkthrough from installation to first inference
- **[Building guide](docs/building.md)** - Build options, hardware acceleration, and troubleshooting
- **[API guide](docs/api-guide.md)** - Detailed usage examples, configuration options, and thread safety

### Quick references

- **[Examples](examples/README.md)** - CLI example with build instructions and usage options
- [Go package documentation](https://pkg.go.dev/github.com/tcpipuk/go-llama.cpp) - Full API reference
- [RELEASE.md](RELEASE.md) - Release process and compatibility tracking

## Requirements

- Go 1.25+
- Docker (recommended) or C++ compiler with CMake
- Git with submodule support

The library uses the GGUF model format, which is the current standard for llama.cpp. For legacy
GGML format support, use the
[pre-gguf tag](https://github.com/tcpipuk/go-llama.cpp/releases/tag/pre-gguf).

## Architecture

The library bridges Go and C++ using CGO, keeping the heavy computation in llama.cpp's optimised
C++ code whilst providing a clean Go API. This minimises CGO overhead whilst maximising
performance.

Key components:

- `wrapper.cpp`/`wrapper.h` - CGO interface to llama.cpp
- `model.go` - Go API using functional options pattern
- `llama.cpp/` - Git submodule containing upstream llama.cpp

The design uses functional options for configuration, resource management with finalizers, and
streaming callbacks via cgo.Handle for safe Go-C interaction.

## Licence

MIT
