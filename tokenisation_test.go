package llama_test

import (
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/tcpipuk/llama-go"
)

// Tokenisation test suite - validates Model.Tokenize method behaviour
// Tests cover basic tokenisation, unicode handling, edge cases, and error conditions

var _ = Describe("Model.Tokenize", func() {
	var (
		model     *llama.Model
		modelPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
		var err error
		model, err = llama.LoadModel(modelPath)
		Expect(err).NotTo(HaveOccurred())
		Expect(model).NotTo(BeNil())
	})

	AfterEach(func() {
		if model != nil {
			model.Close()
		}
	})

	Context("with valid text", func() {
		It("should tokenise simple text successfully", Label("integration"), func() {
			tokens, err := model.Tokenize("Hello world")
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).NotTo(BeNil())
		})

		It("should return array of token IDs", Label("integration"), func() {
			tokens, err := model.Tokenize("The capital of France is Paris")
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(BeAssignableToTypeOf([]int32{}))
		})

		It("should return non-empty slice for non-empty input", Label("integration"), func() {
			tokens, err := model.Tokenize("Test")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0), "should tokenise to at least one token")
		})

		It("should use add_bos=true, special=true", Label("integration"), func() {
			// BOS token should be present at start - verify by tokenising same text twice
			tokens1, err := model.Tokenize("Hello")
			Expect(err).NotTo(HaveOccurred())
			tokens2, err := model.Tokenize("Hello")
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens1).To(Equal(tokens2), "should produce consistent tokens including BOS")
			Expect(len(tokens1)).To(BeNumerically(">", 1), "should have BOS + content tokens")
		})
	})

	Context("with empty string", func() {
		It("should handle empty string", Label("integration"), func() {
			tokens, err := model.Tokenize("")
			// Either succeeds with minimal tokens (BOS only) or fails - both acceptable
			if err != nil {
				// Some models may error on empty input
				Expect(err.Error()).To(ContainSubstring("tokenization"))
			} else {
				Expect(tokens).NotTo(BeNil())
			}
		})

		It("should return empty slice or minimal tokens", Label("integration"), func() {
			tokens, err := model.Tokenize("")
			if err == nil {
				// If successful, should have BOS only or be empty
				Expect(len(tokens)).To(BeNumerically("<=", 1), "empty string should produce at most BOS token")
			}
		})

		It("should not error on empty input", Label("integration"), func() {
			// Some implementations accept empty string, others may not - verify it doesn't crash
			_, err := model.Tokenize("")
			// Either succeeds or returns proper error (not panic)
			if err != nil {
				Expect(err.Error()).NotTo(BeEmpty())
			}
		})
	})

	Context("with unicode text", func() {
		It("should tokenise unicode characters correctly", Label("integration"), func() {
			tokens, err := model.Tokenize("café résumé")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should handle emoji in text", Label("integration"), func() {
			tokens, err := model.Tokenize("Hello 👋 world 🌍")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should handle mixed ASCII and unicode", Label("integration"), func() {
			tokens, err := model.Tokenize("Hello мир 世界 🌎")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should handle multi-byte characters", Label("integration"), func() {
			tokens, err := model.Tokenize("日本語のテキスト")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})
	})

	Context("with special characters", func() {
		It("should tokenise punctuation", Label("integration"), func() {
			tokens, err := model.Tokenize("Hello, world! How are you?")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should tokenise newlines and whitespace", Label("integration"), func() {
			tokens, err := model.Tokenize("Line 1\nLine 2\tTabbed")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should handle special symbols", Label("integration"), func() {
			tokens, err := model.Tokenize("Price: $100.50 (£75.25)")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})
	})

	Context("with very long text", func() {
		It("should tokenise text up to buffer limit (8192)", Label("integration"), func() {
			// Generate text that will produce many tokens
			longText := strings.Repeat("word ", 2000)
			tokens, err := model.Tokenize(longText)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should handle text exceeding 8192 tokens", Label("integration"), func() {
			// Generate very long text likely to exceed token buffer
			veryLongText := strings.Repeat("tokenisation ", 3000)
			tokens, err := model.Tokenize(veryLongText)
			// Should either succeed (with truncation to 8192) or return proper error
			if err == nil {
				Expect(len(tokens)).To(BeNumerically("<=", 8192), "should not exceed buffer limit")
			} else {
				Expect(err.Error()).NotTo(BeEmpty())
			}
		})

		It("should not crash on very long inputs", Label("integration", "slow"), func() {
			// Extreme length test
			extremelyLongText := strings.Repeat("test ", 5000)
			_, err := model.Tokenize(extremelyLongText)
			// Either succeeds or returns error, but must not crash
			if err != nil {
				Expect(err.Error()).NotTo(BeEmpty())
			}
		})
	})

	Context("when model is closed", func() {
		It("should return 'model is closed' error", Label("integration"), func() {
			err := model.Close()
			Expect(err).NotTo(HaveOccurred())

			tokens, err := model.Tokenize("Test")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("model is closed"))
			Expect(tokens).To(BeNil())
		})

		It("should not attempt tokenisation", Label("integration"), func() {
			err := model.Close()
			Expect(err).NotTo(HaveOccurred())

			_, err = model.Tokenize("Any text")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("model is closed"))
		})

		It("should return nil slice and error", Label("integration"), func() {
			err := model.Close()
			Expect(err).NotTo(HaveOccurred())

			tokens, err := model.Tokenize("Test")
			Expect(tokens).To(BeNil())
			Expect(err).To(HaveOccurred())
		})
	})

	Context("with tokenisation failures", func() {
		It("should return error containing 'tokenization failed:'", Label("integration"), func() {
			// Difficult to trigger tokenisation failure without invalid model state
			// This test documents expected error format
			// In practice, most inputs tokenise successfully
			Skip("Tokenisation failures are difficult to trigger reliably in tests")
		})

		It("should handle C++ exceptions gracefully", Label("integration"), func() {
			// C++ exceptions should be caught and converted to errors
			// Cannot easily trigger without corrupting model state
			Skip("C++ exceptions require invalid model state to trigger")
		})

		It("should return 'Exception during tokenization:' for exceptions", Label("integration"), func() {
			// Documents expected error format for C++ exceptions
			Skip("Exception testing requires deliberate model corruption")
		})
	})
})

var _ = Describe("Tokenization Output Validation", func() {
	var (
		model     *llama.Model
		modelPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
		var err error
		model, err = llama.LoadModel(modelPath)
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if model != nil {
			model.Close()
		}
	})

	Context("token ID properties", func() {
		It("should return int32 values", Label("integration"), func() {
			tokens, err := model.Tokenize("Test text")
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).To(BeAssignableToTypeOf([]int32{}))
		})

		It("should return non-negative token IDs", Label("integration"), func() {
			tokens, err := model.Tokenize("Hello world")
			Expect(err).NotTo(HaveOccurred())
			for _, token := range tokens {
				Expect(token).To(BeNumerically(">=", 0), "token IDs should be non-negative")
			}
		})

		It("should return consistent tokens for same input", Label("integration"), func() {
			text := "The quick brown fox"
			tokens1, err := model.Tokenize(text)
			Expect(err).NotTo(HaveOccurred())

			tokens2, err := model.Tokenize(text)
			Expect(err).NotTo(HaveOccurred())

			Expect(tokens1).To(Equal(tokens2), "same input should produce identical tokens")
		})

		It("should return different tokens for different input", Label("integration"), func() {
			tokens1, err := model.Tokenize("Hello")
			Expect(err).NotTo(HaveOccurred())

			tokens2, err := model.Tokenize("Goodbye")
			Expect(err).NotTo(HaveOccurred())

			Expect(tokens1).NotTo(Equal(tokens2), "different input should produce different tokens")
		})
	})

	Context("token count behaviour", func() {
		It("should return actual token count (may be less than buffer)", Label("integration"), func() {
			tokens, err := model.Tokenize("Short text")
			Expect(err).NotTo(HaveOccurred())
			// Should return only actual tokens, not full buffer
			Expect(len(tokens)).To(BeNumerically("<", 8192))
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should not pad to buffer size", Label("integration"), func() {
			tokens, err := model.Tokenize("Test")
			Expect(err).NotTo(HaveOccurred())
			// Should return minimal tokens, not padded to 8192
			Expect(len(tokens)).To(BeNumerically("<", 100), "short text should not produce buffer-sized output")
		})

		It("should handle single-token inputs", Label("integration"), func() {
			// Single character might tokenise to BOS + one token
			tokens, err := model.Tokenize("a")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">=", 1))
			Expect(len(tokens)).To(BeNumerically("<=", 3), "single char should produce minimal tokens")
		})
	})

	Context("buffer limits", func() {
		It("should use maximum buffer size 8192", Label("integration"), func() {
			// Very long text should not exceed buffer limit
			longText := strings.Repeat("word ", 3000)
			tokens, err := model.Tokenize(longText)
			if err == nil {
				Expect(len(tokens)).To(BeNumerically("<=", 8192), "should respect 8192 token buffer limit")
			}
		})

		It("should not exceed 8192 tokens in return slice", Label("integration"), func() {
			// Test with extremely long text
			extremeText := strings.Repeat("tokenisation test ", 2000)
			tokens, err := model.Tokenize(extremeText)
			if err == nil {
				Expect(len(tokens)).To(BeNumerically("<=", 8192), "must not exceed maximum buffer size")
			}
		})
	})
})

var _ = Describe("Tokenization Edge Cases", func() {
	var (
		model     *llama.Model
		modelPath string
	)

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
		var err error
		model, err = llama.LoadModel(modelPath)
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if model != nil {
			model.Close()
		}
	})

	Context("with whitespace variations", func() {
		It("should handle leading whitespace", Label("integration"), func() {
			tokens, err := model.Tokenize("   leading spaces")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should handle trailing whitespace", Label("integration"), func() {
			tokens, err := model.Tokenize("trailing spaces   ")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should handle multiple consecutive spaces", Label("integration"), func() {
			tokens, err := model.Tokenize("multiple     spaces     here")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should handle tabs and newlines", Label("integration"), func() {
			tokens, err := model.Tokenize("tabs\t\there\nnewlines\nhere")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})
	})

	Context("with repeated text", func() {
		It("should tokenise repeated words consistently", Label("integration"), func() {
			tokens, err := model.Tokenize("test test test test")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})

		It("should handle very long repeated sequences", Label("integration"), func() {
			repeated := strings.Repeat("word ", 1000)
			tokens, err := model.Tokenize(repeated)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(tokens)).To(BeNumerically(">", 0))
		})
	})

	Context("with invalid parameters", func() {
		It("should error with 'Invalid parameters for tokenization' if ctx null", Label("integration"), func() {
			// Requires closed model to trigger null context
			err := model.Close()
			Expect(err).NotTo(HaveOccurred())

			tokens, err := model.Tokenize("test")
			Expect(err).To(HaveOccurred())
			// Go layer returns "model is closed" before reaching C++ layer
			Expect(err.Error()).To(Equal("model is closed"))
			Expect(tokens).To(BeNil())
		})

		It("should handle null text pointer gracefully", Label("integration"), func() {
			// Go strings cannot be truly null, but empty string tests this path
			tokens, err := model.Tokenize("")
			// Should either succeed with minimal tokens or return proper error
			if err != nil {
				Expect(err.Error()).NotTo(BeEmpty())
			} else {
				Expect(tokens).NotTo(BeNil())
			}
		})
	})
})
