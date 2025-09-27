# [![Go Reference](https://pkg.go.dev/badge/github.com/tcpipuk/go-llama.cpp.svg)](https://pkg.go.dev/github.com/tcpipuk/go-llama.cpp) go-llama.cpp

Go bindings for [llama.cpp](https://github.com/ggerganov/llama.cpp), the popular C++ library for
running large language models locally. This lets you integrate LLM inference directly into Go
applications with hardware acceleration support.

This is an active fork of [go-skynet/go-llama.cpp](https://github.com/go-skynet/go-llama.cpp),
which appears unmaintained. I'm working to keep it current with the latest llama.cpp developments
and Go best practices.

The bindings use CGO to bridge Go and C++, keeping computational work in optimised C++ code for
best performance whilst providing a clean Go API.

## Quick start

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/tcpipuk/go-llama.cpp
cd go-llama.cpp

# Build the native library
make libbinding.a

# Test with an example
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "/path/to/model.gguf" -t 4
```

For systems without C++ compilers (like TrueNAS), see the [Docker workflow](#docker-development).

## Installation

### Requirements

- Go 1.25+
- C++ compiler (GCC/Clang) or Docker
- Git with submodule support

### Setup

The library depends on llama.cpp as a git submodule, so clone with `--recurse-submodules`:

```bash
git clone --recurse-submodules https://github.com/tcpipuk/go-llama.cpp
cd go-llama.cpp
```

If you've already cloned without submodules, initialise them:

```bash
git submodule update --init --recursive
```

Build the native library:

```bash
make libbinding.a
```

### Docker development

For systems without local C++ compilers, use Docker for the build:

```bash
# Build the native library
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/act-runner:ubuntu-rolling \
  bash -c "apt-get update && apt-get install -y cmake build-essential && make libbinding.a"

# Then use golang container for Go operations
docker run --rm -v $(pwd):/workspace -w /workspace golang:1.25.1-trixie \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace go build ."
```

## Hardware acceleration

### OpenBLAS

For CPU acceleration with OpenBLAS:

```bash
BUILD_TYPE=openblas make libbinding.a
CGO_LDFLAGS="-lopenblas" LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD \
  go run -tags openblas ./examples -m "/path/to/model.gguf" -t 4
```

### CUDA (NVIDIA)

For GPU acceleration with CUDA:

```bash
BUILD_TYPE=cublas make libbinding.a
CGO_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64/" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD \
  go run ./examples -m "/path/to/model.gguf" -t 4
```

### ROCm (AMD)

For AMD GPU acceleration:

```bash
BUILD_TYPE=hipblas make libbinding.a
CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ \
  CGO_LDFLAGS="-O3 --hip-link --rtlib=compiler-rt -unwindlib=libgcc -lrocblas -lhipblas" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD \
  go run ./examples -m "/path/to/model.gguf" -ngl 64 -t 32
```

### Metal (Apple Silicon)

For Apple Silicon acceleration:

```bash
BUILD_TYPE=metal make libbinding.a
CGO_LDFLAGS="-framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go build ./examples/main.go
cp build/bin/ggml-metal.metal .
./main -m "/path/to/model.gguf" -t 1 -ngl 1
```

### OpenCL

For OpenCL acceleration:

```bash
BUILD_TYPE=clblas CLBLAS_DIR=... make libbinding.a
CGO_LDFLAGS="-lOpenCL -lclblast -L/usr/local/lib64/" \
  LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD \
  go run ./examples -m "/path/to/model.gguf" -t 4
```

You should see output like this when OpenCL is working:

```text
ggml_opencl: selecting platform: 'Intel(R) OpenCL HD Graphics'
ggml_opencl: selecting device: 'Intel(R) Graphics [0x46a6]'
ggml_opencl: device FP16 support: true
```

## Usage

The API uses functional options for configuration:

```go
package main

import (
    "fmt"
    llama "github.com/tcpipuk/go-llama.cpp"
)

func main() {
    // Load model with options
    model, err := llama.New(
        "/path/to/model.gguf",
        llama.EnableF16Memory,
        llama.SetContext(2048),
        llama.SetGPULayers(32),
    )
    if err != nil {
        panic(err)
    }
    defer model.Free()

    // Generate text
    response, err := model.Predict("Hello world", llama.SetTokens(50))
    if err != nil {
        panic(err)
    }

    fmt.Println(response)
}
```

Remember to set the CGO environment variables when running:

```bash
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run your-program.go
```

## Architecture

The library bridges Go and C++ using CGO. Most computation stays in the optimised llama.cpp C++
code, with Go providing a clean API layer. This approach minimises CGO overhead whilst maximising
performance.

Key components:

- `binding.cpp` / `binding.h` - CGO interface to llama.cpp
- `llama.go` - Main Go API with the LLama struct
- `options.go` - Configuration using functional options pattern
- `llama.cpp/` - Git submodule containing upstream llama.cpp

## Development

### Testing

Run tests (requires a GGUF model file):

```bash
# Set test model and run
TEST_MODEL="/path/to/test.gguf" make test

# Or manually with Ginkgo
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD \
  go run github.com/onsi/ginkgo/v2/ginkgo --label-filter="!gpu" -v ./...

# In Docker
docker run --rm -v $(pwd):/workspace -w /workspace golang:1.25.1-trixie \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace \
    go run github.com/onsi/ginkgo/v2/ginkgo --label-filter=\"!gpu\" -v ./..."
```

### Code quality

```bash
# Format and vet
go fmt ./...
go vet ./...

# Run pre-commit checks
prek run --all-files
```

### Cleaning

```bash
make clean
```

## Compatibility

This library uses the GGUF model format, which is the standard for llama.cpp. If you need the
legacy GGML format, use the
[pre-gguf tag](https://github.com/tcpipuk/go-llama.cpp/releases/tag/pre-gguf).

## Documentation

Full API documentation is available at
[pkg.go.dev](https://pkg.go.dev/github.com/tcpipuk/go-llama.cpp).

Example code is in [examples/main.go](https://github.com/tcpipuk/go-llama.cpp/blob/master/examples/main.go).

## Licence

MIT
