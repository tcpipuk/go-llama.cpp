# Examples

This directory contains multiple examples demonstrating different llama-go features:

- **`simple/`** - Basic text generation with command-line interface
- **`streaming/`** - Real-time token streaming during generation
- **`speculative/`** - Speculative sampling with draft model acceleration
- **`embedding/`** - Text embedding generation

Each example can be run independently to explore specific functionality.

## Getting started

Start by building the library from the project root:

```bash
# Build using project containers (includes cmake and build tools)
docker run --rm -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"
```

Next, download a test model. We'll use [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF)
as it's small enough to run efficiently even with CPU-only inference:

```bash
# Download test model (~600MB)
wget -q https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf
```

Now you can run the examples:

```bash
# Simple example - single-shot mode
docker run --rm -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples/simple -m Qwen3-0.6B-Q8_0.gguf -p 'Hello world' -n 50"

# Simple example - interactive mode (omit -p flag)
docker run --rm -it -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples/simple -m Qwen3-0.6B-Q8_0.gguf"

# Streaming example
echo "Tell me a story about robots" | docker run --rm -i -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples/streaming -m Qwen3-0.6B-Q8_0.gguf"

# Embedding example
docker run --rm -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples/embedding -m Qwen3-0.6B-Q8_0.gguf -t 'Hello world'"
```

## Simple example options

The `simple/` example accepts these command-line flags:

| Flag | Description | Default |
|------|-------------|---------|
| `-m` | Path to GGUF model file | `./Qwen3-0.6B-Q8_0.gguf` |
| `-p` | Prompt for single-shot mode (interactive if empty) | `"The capital of France is"` |
| `-c` | Context length | `2048` |
| `-ngl` | Number of GPU layers to utilise | `0` |
| `-n` | Number of tokens to predict | `50` |
| `-s` | Predict RNG seed (-1 for random) | `-1` |
| `-temp` | Temperature for sampling | `0.7` |
| `-top-p` | Top-p for sampling | `0.95` |
| `-top-k` | Top-k for sampling | `40` |
| `-debug` | Enable debug output | `false` |

## Example types

### Simple example (`./examples/simple`)

**Single-shot mode:** Use `-p` to provide a prompt and generate text once:

```bash
go run ./examples/simple -m model.gguf -p "The capital of France is" -n 50
```

**Interactive mode:** Omit `-p` to start interactive chat:

```bash
go run ./examples/simple -m model.gguf
```

In interactive mode, type prompts and press Enter twice to submit.

### Streaming example (`./examples/streaming`)

Demonstrates real-time token streaming. Reads from stdin:

```bash
echo "Tell me about AI" | go run ./examples/streaming -m model.gguf
```

### Speculative example (`./examples/speculative`)

Shows speculative sampling with a draft model for faster generation:

```bash
go run ./examples/speculative -target large-model.gguf -draft small-model.gguf -p "Write a story"
```

### Embedding example (`./examples/embedding`)

Generates embeddings for input text:

```bash
go run ./examples/embedding -m embedding-model.gguf -t "Hello world"
```

## Environment variables

These environment variables are required when building and running:

```bash
export LIBRARY_PATH=$PWD        # For linking during build
export C_INCLUDE_PATH=$PWD      # For header files during build
export LD_LIBRARY_PATH=$PWD     # For shared libraries at runtime
```

## Hardware acceleration

To enable GPU acceleration, use the CUDA build container:

```bash
# CUDA build
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"

# CUDA inference with GPU layers
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
  git.tomfos.tr/tom/llama-go:build-cuda \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples/simple -m model.gguf -ngl 32 -p 'Hello world' -n 50"
```

See the [building guide](../docs/building.md) for all hardware acceleration options.
