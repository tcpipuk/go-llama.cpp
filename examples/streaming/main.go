// Streaming example demonstrates token-by-token generation using callbacks.
//
// This interactive program loads a GGUF model and accepts multi-line prompts,
// streaming generated tokens to stdout as they're produced. This showcases
// real-time response generation suitable for chat interfaces or interactive
// applications.
//
// Usage:
//
//	streaming -m model.gguf
//
// The program prompts for input interactively. Enter your prompt on one or
// more lines, then submit with an empty line. Generated text streams to stdout
// token-by-token as the model produces it.
//
// The example demonstrates:
//   - Streaming generation with callbacks
//   - Interactive multi-line input handling
//   - Real-time token output
//   - Stop word configuration for controlled responses
//   - Graceful error handling during generation
//
// This pattern is ideal for building chat applications, REPLs, or any interface
// requiring immediate feedback during generation.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"

	llama "github.com/tcpipuk/llama-go"
)

func main() {
	var (
		modelPath = flag.String("m", "./Qwen3-0.6B-Q8_0.gguf", "path to GGUF model file")
		maxTokens = flag.Int("n", 512, "maximum number of tokens to generate")
		context   = flag.Int("c", 2048, "context size")
		gpuLayers = flag.Int("ngl", -1, "number of GPU layers (-1 for all)")
		temp      = flag.Float64("temp", 0.8, "temperature for sampling")
		topP      = flag.Float64("top-p", 0.9, "top-p for sampling")
		topK      = flag.Int("top-k", 40, "top-k for sampling")
	)
	flag.Parse()

	// Load model
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
	fmt.Println("Interactive streaming mode - enter prompts (empty line to submit, Ctrl+C to exit)")

	reader := bufio.NewReader(os.Stdin)

	for {
		prompt := readMultiLineInput(reader)
		if prompt == "" {
			continue
		}

		fmt.Printf("\nGenerating response:\n")

		// Generate with streaming callback
		err := model.GenerateStream(prompt,
			func(token string) bool {
				fmt.Print(token)
				return true // Continue generation
			},
			llama.WithMaxTokens(*maxTokens),
			llama.WithTemperature(float32(*temp)),
			llama.WithTopP(float32(*topP)),
			llama.WithTopK(*topK),
			llama.WithStopWords("Human:", "Assistant:", "\n\n"),
		)

		if err != nil {
			fmt.Fprintf(os.Stderr, "\nError generating text: %v\n", err)
		}

		fmt.Printf("\n\n--- End of response ---\n\n")
	}
}

// readMultiLineInput reads input until an empty line is entered
func readMultiLineInput(reader *bufio.Reader) string {
	var lines []string
	fmt.Print(">>> ")

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if strings.Contains(err.Error(), "EOF") {
				os.Exit(0)
			}
			fmt.Printf("Error reading input: %v\n", err)
			continue
		}

		if len(strings.TrimSpace(line)) == 0 {
			break
		}

		lines = append(lines, line)
	}

	text := strings.Join(lines, "")
	fmt.Printf("Sending: %s\n", strings.TrimSpace(text))
	return text
}
