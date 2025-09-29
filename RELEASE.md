# Release process

This document outlines how to create new releases of go-llama.cpp that maintain
compatibility with upstream llama.cpp development.

## Overview

Rather than distributing pre-built binaries, this fork tracks specific llama.cpp
versions so Go developers can build exactly what they need. Each release corresponds
to a tested, compatible upstream version - no guesswork about which llama.cpp
features are available.

## Release workflow

### 1. Monitor upstream releases

Check [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) for new
stable releases. Focus on releases that introduce API changes, performance
improvements, or new model support.

### 2. Clean existing build artifacts

Before updating, ensure a clean build environment:

```bash
# Check existing artifacts
ls -la *.a *.so 2>/dev/null

# Clean all build artifacts using Docker for consistent environment
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/act-runner:ubuntu-rolling \
  make clean

# Verify cleanup - should return no results
ls -la *.a *.so 2>/dev/null
ls -la llama.cpp/*.o 2>/dev/null
```

**Verification**: Ensure no `.a`, `.so`, or `.o` files remain before proceeding.

### 3. Update the submodule

Update the llama.cpp submodule to the target release:

```bash
# Check submodule status first
cd llama.cpp
git status  # Should be clean before proceeding

# Fetch and checkout target release
git fetch origin
git checkout <target-tag>  # e.g. b6615

# Verify correct tag
git describe --tags  # Should show exact tag

# Ensure clean state (no modifications)
git status  # Should show "HEAD detached at <tag>" with no changes

cd ..

# Check diff before staging to prevent "-dirty" commits
git diff llama.cpp  # Should show clean pointer update
git submodule status  # Should show no + prefix (clean)

# Stage the submodule update
git add llama.cpp

# Verify staged change
git diff --cached  # Should show clean submodule pointer update
```

**Verification**: The submodule must be at the exact tag with no local modifications.

### 4. Test compatibility

Build and test the updated bindings:

```bash
# Verify no existing build artifacts
ls -la libbinding.a 2>/dev/null  # Should return nothing

# Build with Docker for consistent environment
docker run --rm -v $(pwd):/workspace -w /workspace git.tomfos.tr/tom/act-runner:ubuntu-rolling \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace make libbinding.a"

# Verify build created all artifacts
ls -la libbinding.a libcommon.a *.so
ls -lat *.a *.so | head -10  # Check timestamps confirm fresh build

# Download test models if not present
ls -la Qwen3-0.6B-Q8_0.gguf 2>/dev/null || \
  wget -q https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf

# Download embedding test model for embedding functionality
ls -la Qwen3-Embedding-0.6B-Q8_0.gguf 2>/dev/null || \
  wget -q https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf

# Test inference pipeline
docker run --rm -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples -m Qwen3-0.6B-Q8_0.gguf -p 'Hello world' -n 50"

# Test embedding functionality
docker run --rm -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           go run ./examples/embedding -m Qwen3-Embedding-0.6B-Q8_0.gguf -t 'Hello world'"

# Run test suite with test model (tests inference, speculative sampling, tokenisation)
docker run --rm -v $(pwd):/workspace -w /workspace golang:latest \
  bash -c "LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
           TEST_MODEL=Qwen3-0.6B-Q8_0.gguf go run github.com/onsi/ginkgo/v2/ginkgo -v ./..."
```

**Verification**: All commands must complete successfully with expected output before proceeding.

**Note**: The test suite includes a GPU test that validates GPU fallback behaviour
(graceful degradation to CPU when no GPU is available). For actual GPU acceleration
testing, see the GPU testing section below.

### GPU Acceleration Testing (Optional)

For projects requiring GPU acceleration, test CUDA support:

#### Building with CUDA support

1. Build using the CUDA Docker container:

   ```bash
   docker build -f Dockerfile.cuda -t go-llama-cuda .

   # Build for specific GPU architecture (e.g., RTX 3090)
   docker run --rm --gpus all -v $(pwd):/workspace go-llama-cuda \
     bash -c "CUDA_ARCHITECTURES=86 BUILD_TYPE=cublas make libbinding.a"
   ```

2. Verify CUDA libraries are built:

   ```bash
   ls -la *.so | grep ggml
   strings libggml.so | grep -i cuda | head -5  # Should show cuda symbols
   ```

3. Test GPU acceleration with Go 1.25+:

   ```bash
   docker run --rm --gpus all -v $(pwd):/workspace go-llama-cuda \
     bash -c "wget -q https://go.dev/dl/go1.25.1.linux-amd64.tar.gz && \
              tar -C /usr/local -xzf go1.25.1.linux-amd64.tar.gz && \
              export PATH=/usr/local/go/bin:\$PATH && \
              cd /workspace && \
              LIBRARY_PATH=/workspace C_INCLUDE_PATH=/workspace LD_LIBRARY_PATH=/workspace \
              TEST_MODEL=Qwen3-0.6B-Q8_0.gguf \
              go run github.com/onsi/ginkgo/v2/ginkgo --label-filter='gpu' -v ./..."
   ```

**Expected output**:

- "found 1 CUDA devices: Device 0: NVIDIA GeForce RTX 3090"
- "offloaded X/Y layers to GPU"
- "CUDA0 model buffer size = XXX MiB"

#### Common GPU architectures

| GPU Series | Architecture | Code |
|------------|--------------|------|
| GTX 1000 (Pascal) | 6.1 | 61 |
| RTX 2000 (Turing) | 7.5 | 75 |
| RTX 3000 (Ampere) | 8.6 | 86 |
| RTX 4000 (Ada) | 8.9 | 89 |
| A100 (Ampere) | 8.0 | 80 |

Build for your specific GPU for faster compilation and smaller binaries:

```bash
CUDA_ARCHITECTURES=86 BUILD_TYPE=cublas make libbinding.a  # RTX 3090
```

### 5. Fix any API compatibility issues

If the build fails or tests don't pass, update the Go bindings:

- Check `binding.cpp` for API changes
- Update header includes if files have moved
- Adjust function calls for parameter changes
- Test again until inference works

### 6. Commit and tag the release

Once compatibility is confirmed:

```bash
git commit -m "feat(deps): update llama.cpp to <tag> and fix API compatibility"
git tag -a llama.cpp-<tag> -m "Update to llama.cpp <tag> with API compatibility fixes"
git push origin main --tags
```

## Why no pre-built binaries

Unlike traditional Go libraries, this project doesn't distribute pre-built static
libraries. The reason is simple: there are too many build variants to support
effectively.

Different users need different configurations - CPU-only for development, CUDA for
NVIDIA cards, ROCm for AMD, Metal for Apple Silicon, or OpenBLAS for optimised CPU
inference. Each acceleration backend requires different compiler flags,
hardware-specific SDKs, and runtime dependencies that vary by system.

Instead, applications using this library build their own variants as part of their
deployment pipeline. This lets them choose appropriate acceleration for their
hardware, include the right runtime dependencies, and cache builds for their
specific configuration.

## Versioning convention

Tags follow the format `llama.cpp-{upstream-tag}` to clearly indicate compatibility:

- `llama.cpp-b6603` - Compatible with llama.cpp tag b6603
- `llama.cpp-b6580` - Compatible with llama.cpp tag b6580

This makes it easy for consumers to choose compatible versions, understand which
llama.cpp features are available, and track upstream development without confusion.

## Testing checklist

Before tagging a release, verify:

- [ ] All existing build artifacts removed via `make clean`
- [ ] No `.a`, `.so`, or `.o` files present before build
- [ ] Submodule updated to target llama.cpp release
- [ ] Submodule has no local modifications (clean checkout)
- [ ] Library builds successfully with Docker containers
- [ ] All expected artifacts created with fresh timestamps
- [ ] Example program loads test model without errors
- [ ] Text generation produces reasonable output
- [ ] Test suite passes without failures
- [ ] No obvious API compatibility warnings or errors
- [ ] Commit shows clean submodule pointer update (no "-dirty" suffix)
- [ ] Commit includes clear description of changes made
