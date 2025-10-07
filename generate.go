package llama

// Generate generates text from the given prompt.
//
// This method performs synchronous text generation, returning the complete
// result when finished. The library automatically reuses KV cache entries for
// matching prompt prefixes (prefix caching), significantly improving performance
// for conversation-style usage.
//
// Examples:
//
//	// Basic generation
//	result, err := model.Generate("Once upon a time")
//
//	// With custom parameters
//	result, err := model.Generate("Explain quantum physics",
//	    llama.WithMaxTokens(512),
//	    llama.WithTemperature(0.7),
//	    llama.WithStopWords("\n\n"),
//	)
//
//	// Conversation with prefix caching
//	sys := "You are a helpful assistant.\n\n"
//	model.Generate(sys + "User: Hello")
//	model.Generate(sys + "User: How are you?")  // Reuses cached system prompt
func (m *Model) Generate(prompt string, opts ...GenerateOption) (string, error) {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return m.generateWithConfig(prompt, config, nil)
}

// GenerateStream generates text with streaming output via callback.
//
// The callback receives each generated token as it's produced. Return true to
// continue generation, or false to stop early. The callback executes
// synchronously within the generation loop, so expensive operations will block
// token generation.
//
// Examples:
//
//	// Stream to stdout
//	err := model.GenerateStream("Tell me a story",
//	    func(token string) bool {
//	        fmt.Print(token)
//	        return true
//	    },
//	)
//
//	// Stop early based on content
//	err := model.GenerateStream("Count to 100",
//	    func(token string) bool {
//	        fmt.Print(token)
//	        return !strings.Contains(token, "50")  // Stop at 50
//	    },
//	)
func (m *Model) GenerateStream(prompt string, callback func(token string) bool, opts ...GenerateOption) error {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	_, err := m.generateWithConfig(prompt, config, callback)
	return err
}

// GenerateWithDraft performs speculative generation using a draft model.
//
// Speculative decoding uses a smaller draft model to generate candidate tokens
// that the target model verifies in parallel. This reduces latency whilst
// maintaining the target model's quality. The draft model should be
// significantly smaller (e.g. 1-3B vs 70B) for best speedup.
//
// Examples:
//
//	target, _ := llama.LoadModel("large-model.gguf")
//	draft, _ := llama.LoadModel("small-model.gguf")
//	defer target.Close()
//	defer draft.Close()
//
//	result, err := target.GenerateWithDraft("Once upon a time", draft,
//	    llama.WithDraftTokens(8),
//	    llama.WithMaxTokens(256),
//	)
func (m *Model) GenerateWithDraft(prompt string, draft *Model, opts ...GenerateOption) (string, error) {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	return m.generateWithDraftAndConfig(prompt, draft, config, nil)
}

// GenerateWithDraftStream performs speculative generation with streaming output.
//
// Combines speculative decoding with token-by-token streaming. The callback
// receives each verified token as it's accepted by the target model. See
// GenerateWithDraft for speculative decoding details and GenerateStream for
// callback semantics.
//
// Example:
//
//	target, _ := llama.LoadModel("large-model.gguf")
//	draft, _ := llama.LoadModel("small-model.gguf")
//
//	err := target.GenerateWithDraftStream("Write a story", draft,
//	    func(token string) bool {
//	        fmt.Print(token)
//	        return true
//	    },
//	    llama.WithDraftTokens(8),
//	)
func (m *Model) GenerateWithDraftStream(prompt string, draft *Model, callback func(token string) bool, opts ...GenerateOption) error {
	config := defaultGenerateConfig
	for _, opt := range opts {
		opt(&config)
	}

	_, err := m.generateWithDraftAndConfig(prompt, draft, config, callback)
	return err
}
