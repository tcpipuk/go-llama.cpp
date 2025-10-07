package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	llama "github.com/tcpipuk/llama-go"
)

func main() {
	var (
		modelPath = flag.String("m", "./Qwen3-0.6B-Q8_0.gguf", "path to GGUF model file")
		prompt    = flag.String("p", "The capital of France is", "prompt for generation")
		maxTokens = flag.Int("n", 50, "maximum number of tokens to generate")
		context   = flag.Int("c", 2048, "context size")
		gpuLayers = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		temp      = flag.Float64("temp", 0.7, "temperature for sampling")
		topP      = flag.Float64("top-p", 0.95, "top-p for sampling")
		topK      = flag.Int("top-k", 40, "top-k for sampling")
		seed      = flag.Int("s", -1, "random seed")
		debug     = flag.Bool("debug", false, "enable debug output")
	)
	flag.Parse()

	// Load model with options
	fmt.Printf("Loading model: %s\n", *modelPath)
	model, err := llama.LoadModel(*modelPath,
		llama.WithContext(*context),
		llama.WithGPULayers(*gpuLayers),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithF16Memory(),
		llama.WithMMap(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}
	defer model.Close()

	fmt.Printf("Model loaded successfully.\n")
	fmt.Printf("Prompt: %s\n", *prompt)

	// Generate text
	var opts []llama.GenerateOption
	opts = append(opts, llama.WithMaxTokens(*maxTokens))
	opts = append(opts, llama.WithTemperature(float32(*temp)))
	opts = append(opts, llama.WithTopP(float32(*topP)))
	opts = append(opts, llama.WithTopK(*topK))
	opts = append(opts, llama.WithSeed(*seed))
	if *debug {
		opts = append(opts, llama.WithDebug())
	}

	response, err := model.Generate(*prompt, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating text: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nResponse: %s\n", response)
}
