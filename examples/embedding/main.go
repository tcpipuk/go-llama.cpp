package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	llama "github.com/tcpipuk/go-llama.cpp"
)

func main() {
	var (
		modelPath = flag.String("m", "./Qwen3-Embedding-0.6B-Q8_0.gguf", "path to GGUF embedding model file")
		text      = flag.String("t", "Hello world", "text to get embeddings for")
		gpuLayers = flag.Int("ngl", 0, "number of GPU layers")
		context   = flag.Int("c", 2048, "context size")
	)
	flag.Parse()

	// Load model with embeddings enabled
	fmt.Printf("Loading embedding model: %s\n", *modelPath)
	model, err := llama.LoadModel(*modelPath,
		llama.WithContext(*context),
		llama.WithGPULayers(*gpuLayers),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithEmbeddings(),
		llama.WithF16Memory(),
		llama.WithMMap(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	fmt.Printf("Model loaded successfully.\n")
	fmt.Printf("Getting embeddings for: %s\n", *text)

	// Generate embeddings
	embeddings, err := model.GetEmbeddings(*text)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating embeddings: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nEmbeddings generated successfully!\n")
	fmt.Printf("Vector dimension: %d\n", len(embeddings))
	fmt.Printf("First 10 values: ")
	for i := 0; i < 10 && i < len(embeddings); i++ {
		fmt.Printf("%.4f ", embeddings[i])
	}
	fmt.Printf("\n")

	// Calculate magnitude for demonstration
	magnitude := float32(0.0)
	for _, val := range embeddings {
		magnitude += val * val
	}
	magnitude = float32(1.0) / float32(len(embeddings)) * magnitude // Mean squared
	fmt.Printf("Mean squared magnitude: %.6f\n", magnitude)
}
