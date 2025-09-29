package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"

	llama "github.com/tcpipuk/go-llama.cpp"
)

func main() {
	var (
		modelPath = flag.String("m", "./Qwen3-0.6B-Q8_0.gguf", "path to GGUF model file")
		maxTokens = flag.Int("n", 512, "maximum number of tokens to generate")
		context   = flag.Int("c", 2048, "context size")
		gpuLayers = flag.Int("ngl", 0, "number of GPU layers")
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
