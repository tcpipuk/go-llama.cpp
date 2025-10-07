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
		targetModel = flag.String("target", "./Qwen3-0.6B-Q8_0.gguf", "path to target (main) model")
		draftModel  = flag.String("draft", "./Qwen3-0.6B-Q8_0.gguf", "path to draft model")
		prompt      = flag.String("p", "The capital of France is", "prompt for generation")
		maxTokens   = flag.Int("n", 100, "maximum number of tokens to generate")
		context     = flag.Int("c", 2048, "context size")
		gpuLayers   = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		temp        = flag.Float64("temp", 0.7, "temperature for sampling")
		draftTokens = flag.Int("draft-tokens", 16, "number of draft tokens per iteration")
		debug       = flag.Bool("debug", false, "enable debug output")
	)
	flag.Parse()

	// Load target model
	fmt.Printf("Loading target model: %s\n", *targetModel)
	target, err := llama.LoadModel(*targetModel,
		llama.WithContext(*context),
		llama.WithGPULayers(*gpuLayers),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithF16Memory(),
		llama.WithMMap(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading target model: %v\n", err)
		os.Exit(1)
	}
	defer target.Close()

	// Load draft model
	fmt.Printf("Loading draft model: %s\n", *draftModel)
	draft, err := llama.LoadModel(*draftModel,
		llama.WithContext(*context),
		llama.WithGPULayers(*gpuLayers),
		llama.WithThreads(runtime.NumCPU()),
		llama.WithF16Memory(),
		llama.WithMMap(true),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading draft model: %v\n", err)
		os.Exit(1)
	}
	defer draft.Close()

	fmt.Printf("Models loaded successfully.\n")
	fmt.Printf("Prompt: %s\n", *prompt)
	fmt.Printf("Using speculative sampling with %d draft tokens per iteration\n", *draftTokens)

	// Generate with speculative sampling
	var opts []llama.GenerateOption
	opts = append(opts, llama.WithMaxTokens(*maxTokens))
	opts = append(opts, llama.WithTemperature(float32(*temp)))
	opts = append(opts, llama.WithDraftTokens(*draftTokens))
	if *debug {
		opts = append(opts, llama.WithDebug())
	}

	response, err := target.GenerateWithDraft(*prompt, draft, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error generating text: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nResponse: %s\n", response)
}
