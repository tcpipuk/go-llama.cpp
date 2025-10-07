package llama

// Generation options

// WithMaxTokens sets the maximum number of tokens to generate.
//
// Generation stops after producing this many tokens, even if the model hasn't
// emitted an end-of-sequence token. This prevents runaway generation and
// controls response length.
//
// Default: 128
//
// Example:
//
//	// Generate up to 512 tokens
//	text, err := model.Generate("Write a story",
//	    llama.WithMaxTokens(512),
//	)
func WithMaxTokens(n int) GenerateOption {
	return func(c *generateConfig) {
		c.maxTokens = n
	}
}

// WithTemperature controls randomness in token selection.
//
// Higher values (e.g. 1.2) increase creativity and diversity but may reduce
// coherence. Lower values (e.g. 0.3) make output more deterministic and
// focused. Use 0.0 for fully deterministic greedy sampling (always pick the
// most likely token).
//
// Default: 0.8
//
// Examples:
//
//	// Creative writing
//	text, err := model.Generate("Write a poem",
//	    llama.WithTemperature(1.1),
//	)
//
//	// Precise factual responses
//	text, err := model.Generate("What is 2+2?",
//	    llama.WithTemperature(0.1),
//	)
func WithTemperature(t float32) GenerateOption {
	return func(c *generateConfig) {
		c.temperature = t
	}
}

// WithTopP enables nucleus sampling with the specified cumulative probability.
//
// Top-p sampling (nucleus sampling) considers only the smallest set of tokens
// whose cumulative probability exceeds p. This balances diversity and quality
// better than top-k for many tasks. Use 1.0 to disable (consider all tokens).
//
// Default: 0.95
//
// Example:
//
//	// More focused sampling
//	text, err := model.Generate("Complete this",
//	    llama.WithTopP(0.85),
//	)
func WithTopP(p float32) GenerateOption {
	return func(c *generateConfig) {
		c.topP = p
	}
}

// WithTopK limits token selection to the k most likely candidates.
//
// Top-k sampling considers only the k highest probability tokens at each step.
// Lower values increase focus and determinism, higher values increase diversity.
// Use 0 to disable (consider all tokens).
//
// Default: 40
//
// Example:
//
//	// Very focused generation
//	text, err := model.Generate("Complete this",
//	    llama.WithTopK(10),
//	)
func WithTopK(k int) GenerateOption {
	return func(c *generateConfig) {
		c.topK = k
	}
}

// WithSeed sets the random seed for reproducible generation.
//
// Using the same seed with identical settings produces deterministic output.
// Use -1 for random seed (different output each time). Useful for testing,
// debugging, or when reproducibility is required.
//
// Default: -1 (random)
//
// Example:
//
//	// Reproducible generation
//	text, err := model.Generate("Write a story",
//	    llama.WithSeed(42),
//	    llama.WithTemperature(0.8),
//	)
func WithSeed(seed int) GenerateOption {
	return func(c *generateConfig) {
		c.seed = seed
	}
}

// WithStopWords specifies sequences that terminate generation when encountered.
//
// Generation stops immediately when any stop word is produced. Useful for
// controlling response format (e.g. stopping at newlines) or implementing
// chat patterns. The stop words themselves are not included in the output.
//
// Default: none
//
// Examples:
//
//	// Stop at double newline
//	text, err := model.Generate("Q: What is AI?",
//	    llama.WithStopWords("\n\n"),
//	)
//
//	// Multiple stop sequences
//	text, err := model.Generate("User:",
//	    llama.WithStopWords("User:", "Assistant:", "\n\n"),
//	)
func WithStopWords(words ...string) GenerateOption {
	return func(c *generateConfig) {
		c.stopWords = words
	}
}

// WithDraftTokens sets the number of speculative tokens for draft model usage.
//
// When using GenerateWithDraft, the draft model speculatively generates this
// many tokens per iteration. Higher values increase potential speedup but
// waste more work if predictions are rejected. Typical range: 4-32 tokens.
//
// Default: 16
//
// Example:
//
//	target, _ := llama.LoadModel("large-model.gguf")
//	draft, _ := llama.LoadModel("small-model.gguf")
//	text, err := target.GenerateWithDraft("Write a story", draft,
//	    llama.WithDraftTokens(8),
//	)
func WithDraftTokens(n int) GenerateOption {
	return func(c *generateConfig) {
		c.draftTokens = n
	}
}

// WithDebug enables verbose logging for generation internals.
//
// When enabled, prints detailed information about token sampling, timing,
// and internal state to stderr. Useful for debugging generation issues or
// understanding model behaviour. Not recommended for production use.
//
// Default: false
//
// Example:
//
//	text, err := model.Generate("Test prompt",
//	    llama.WithDebug(),
//	)
func WithDebug() GenerateOption {
	return func(c *generateConfig) {
		c.debug = true
	}
}

// WithPrefixCaching enables or disables KV cache prefix reuse.
// When enabled (default), the library reuses KV cache entries for matching
// prompt prefixes, significantly improving performance for conversation-style
// usage or repeated prompts.
//
// Disable this if you need guaranteed-fresh state for every generation,
// though the performance cost is minimal with the last-token refresh optimisation.
//
// Default: true (enabled)
func WithPrefixCaching(enabled bool) GenerateOption {
	return func(c *generateConfig) {
		c.prefixCaching = enabled
	}
}
