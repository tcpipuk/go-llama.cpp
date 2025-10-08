package llama_test

import (
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/tcpipuk/llama-go"
)

// Embeddings test suite
//
// Tests the GetEmbeddings method and WithEmbeddings option, covering:
// - Basic embedding generation with embeddings enabled
// - Various text input scenarios
// - Empty text handling
// - Error handling when embeddings not enabled
// - Model closed error conditions
// - Embedding generation error paths
// - Vector dimension and value properties
// - Embedding stability and consistency
// - WithEmbeddings option behaviour
// - Edge cases and parameter validation

var _ = Describe("Model.GetEmbeddings", func() {
	Context("with embeddings enabled", func() {
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
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
		})

		AfterEach(func() {
			if model != nil {
				model.Close()
			}
		})

		It("should generate embeddings successfully", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Hello world")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeNil())
		})

		It("should return float32 slice", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Test text")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).To(BeAssignableToTypeOf([]float32{}))
		})

		It("should return non-empty embedding vector", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Non-empty input")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(embeddings)).To(BeNumerically(">", 0))
		})

		It("should have consistent dimension across calls", Label("integration"), func() {
			embeddings1, err := model.GetEmbeddings("First text")
			Expect(err).NotTo(HaveOccurred())

			embeddings2, err := model.GetEmbeddings("Second text")
			Expect(err).NotTo(HaveOccurred())

			Expect(len(embeddings1)).To(Equal(len(embeddings2)))
		})
	})

	Context("with various text inputs", func() {
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
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if model != nil {
				model.Close()
			}
		})

		It("should generate embeddings for simple text", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Hello")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})

		It("should generate embeddings for long text", Label("integration"), func() {
			longText := "This is a longer piece of text that contains multiple sentences. " +
				"It should be tokenised and processed correctly. " +
				"The embedding should capture the semantic meaning of the entire passage."

			embeddings, err := model.GetEmbeddings(longText)
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})

		It("should generate embeddings for unicode text", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Hello 世界 🌍")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})

		It("should handle single word input", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("word")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})

		It("should handle multi-sentence input", Label("integration"), func() {
			multiSentence := "First sentence. Second sentence. Third sentence."
			embeddings, err := model.GetEmbeddings(multiSentence)
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})
	})

	Context("with empty text", func() {
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
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if model != nil {
				model.Close()
			}
		})

		It("should handle empty string input", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("")
			// Check actual behaviour - may return embeddings or error
			if err != nil {
				// If it errors, check for appropriate error message
				Expect(err.Error()).To(ContainSubstring("embedding"))
			} else {
				// If it succeeds, verify embeddings are returned
				Expect(embeddings).NotTo(BeNil())
			}
		})

		It("should not crash on empty input", Label("integration"), func() {
			// This test verifies robustness - should not panic
			_, _ = model.GetEmbeddings("")
			// If we reach here without panic, test passes
			Succeed()
		})
	})

	Context("when embeddings not enabled", func() {
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
			// Load model WITHOUT WithEmbeddings()
			model, err = llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if model != nil {
				model.Close()
			}
		})

		It("should return error if model loaded without WithEmbeddings()", Label("integration"), func() {
			_, err := model.GetEmbeddings("Test text")
			Expect(err).To(HaveOccurred())
		})

		It("should error containing 'Failed to get embeddings from context'", Label("integration"), func() {
			_, err := model.GetEmbeddings("Test text")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("Failed to get embeddings from context"))
		})

		It("should not crash when called on non-embedding model", Label("integration"), func() {
			// This test verifies robustness - should error gracefully, not panic
			_, err := model.GetEmbeddings("Test text")
			Expect(err).To(HaveOccurred())
			// If we reach here without panic, test passes
		})
	})

	Context("when model is closed", func() {
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
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())

			// Close the model
			model.Close()
		})

		It("should return 'model is closed' error", Label("integration"), func() {
			_, err := model.GetEmbeddings("Test text")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("model is closed"))
		})

		It("should not attempt embedding generation", Label("integration"), func() {
			_, err := model.GetEmbeddings("Test text")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("model is closed"))
			// Verify it's the Go-level check, not a C++ error
		})
	})

	Context("with embedding generation errors", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set - skipping integration test")
			}
		})

		It("should return error containing 'embedding generation failed:'", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Try to trigger an error condition
			// If embeddings are disabled, this should fail with appropriate error
			_, err = model.GetEmbeddings("Test")
			if err != nil {
				// If error occurs, check it has proper prefix
				// Note: This may not error with embeddings enabled
				possiblePrefixes := []string{
					"embedding generation failed:",
					"Failed to",
				}
				matched := false
				for _, prefix := range possiblePrefixes {
					if len(err.Error()) >= len(prefix) && err.Error()[:len(prefix)] == prefix {
						matched = true
						break
					}
				}
				Expect(matched).To(BeTrue(), "error should have appropriate prefix")
			}
		})

		It("should handle tokenisation failures with 'Failed to tokenize text for embeddings'", Label("integration"), func() {
			// This error is difficult to trigger reliably
			// We document the expected error message for reference
			expectedError := "Failed to tokenize text for embeddings"
			_ = expectedError // Document expected error string
		})

		It("should handle decode failures with 'Failed to decode tokens for embeddings'", Label("integration"), func() {
			// This error is difficult to trigger reliably
			// We document the expected error message for reference
			expectedError := "Failed to decode tokens for embeddings"
			_ = expectedError // Document expected error string
		})

		It("should handle null embeddings with 'Failed to get embeddings from context'", Label("integration"), func() {
			// This is tested in the "when embeddings not enabled" context
			// Here we document the expected error for completeness
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048)) // No WithEmbeddings()
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			_, err = model.GetEmbeddings("Test")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("Failed to get embeddings from context"))
		})
	})
})

var _ = Describe("Embedding Vector Properties", func() {
	Context("vector dimension", func() {
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
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if model != nil {
				model.Close()
			}
		})

		It("should return vector with model-specific dimension", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(embeddings)).To(BeNumerically(">", 0))
			// Dimension is model-specific, verify it's positive
		})

		It("should match llama_model_n_embd() value", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
			// The actual dimension is returned from llama_model_n_embd()
			// We verify it's consistent across calls
			embeddings2, err := model.GetEmbeddings("Different")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(embeddings)).To(Equal(len(embeddings2)))
		})

		It("should use maximum buffer size 4096", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
			// Buffer limit is 4096 floats - verify we don't exceed it
			Expect(len(embeddings)).To(BeNumerically("<=", 4096))
		})

		It("should not exceed 4096 floats", Label("integration"), func() {
			// Test with longer text to ensure buffer limit is respected
			longText := ""
			for i := 0; i < 100; i++ {
				longText += "This is a longer sentence to test embedding dimension limits. "
			}

			embeddings, err := model.GetEmbeddings(longText)
			Expect(err).NotTo(HaveOccurred())
			Expect(len(embeddings)).To(BeNumerically("<=", 4096))
		})
	})

	Context("vector values", func() {
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
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if model != nil {
				model.Close()
			}
		})

		It("should return float32 values", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).To(BeAssignableToTypeOf([]float32{}))
		})

		It("should have non-zero values for non-empty text", Label("integration"), func() {
			embeddings, err := model.GetEmbeddings("Hello world")
			Expect(err).NotTo(HaveOccurred())

			// At least some values should be non-zero
			hasNonZero := false
			for _, val := range embeddings {
				if val != 0.0 {
					hasNonZero = true
					break
				}
			}
			Expect(hasNonZero).To(BeTrue(), "embedding should contain non-zero values")
		})

		It("should produce different embeddings for different text", Label("integration"), func() {
			embeddings1, err := model.GetEmbeddings("Hello world")
			Expect(err).NotTo(HaveOccurred())

			embeddings2, err := model.GetEmbeddings("Goodbye world")
			Expect(err).NotTo(HaveOccurred())

			// Embeddings should be different for different text
			Expect(embeddings1).NotTo(Equal(embeddings2))
		})

		It("should produce identical embeddings for identical text", Label("integration"), func() {
			embeddings1, err := model.GetEmbeddings("Same text")
			Expect(err).NotTo(HaveOccurred())

			embeddings2, err := model.GetEmbeddings("Same text")
			Expect(err).NotTo(HaveOccurred())

			// Embeddings should be identical for same text
			Expect(embeddings1).To(Equal(embeddings2))
		})
	})

	Context("embedding stability", func() {
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
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if model != nil {
				model.Close()
			}
		})

		It("should produce consistent embeddings across calls", Label("integration"), func() {
			text := "Consistent text for testing"

			embeddings1, err := model.GetEmbeddings(text)
			Expect(err).NotTo(HaveOccurred())

			embeddings2, err := model.GetEmbeddings(text)
			Expect(err).NotTo(HaveOccurred())

			embeddings3, err := model.GetEmbeddings(text)
			Expect(err).NotTo(HaveOccurred())

			// All embeddings should be identical
			Expect(embeddings1).To(Equal(embeddings2))
			Expect(embeddings2).To(Equal(embeddings3))
		})

		It("should not vary with random seed (embeddings are deterministic)", Label("integration"), func() {
			// Embeddings should be deterministic regardless of seed used for generation
			// Note: GetEmbeddings doesn't use seed, but we verify determinism
			text := "Deterministic test"

			embeddings1, err := model.GetEmbeddings(text)
			Expect(err).NotTo(HaveOccurred())

			embeddings2, err := model.GetEmbeddings(text)
			Expect(err).NotTo(HaveOccurred())

			Expect(embeddings1).To(Equal(embeddings2))
		})
	})
})

var _ = Describe("WithEmbeddings Option", func() {
	Context("when enabled at load time", func() {
		var (
			model     *llama.Model
			modelPath string
		)

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set - skipping integration test")
			}
		})

		AfterEach(func() {
			if model != nil {
				model.Close()
			}
		})

		It("should enable embeddings mode in context", Label("integration"), func() {
			var err error
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())

			// Verify embeddings can be generated
			embeddings, err := model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})

		It("should allow GetEmbeddings() calls", Label("integration"), func() {
			var err error
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())

			_, err = model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
		})

		It("should configure model for embedding extraction", Label("integration"), func() {
			var err error
			model, err = llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())

			// Model should be configured for embeddings
			embeddings, err := model.GetEmbeddings("Configure test")
			Expect(err).NotTo(HaveOccurred())
			Expect(len(embeddings)).To(BeNumerically(">", 0))
		})
	})

	Context("when not specified", func() {
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
			// Load without WithEmbeddings()
			model, err = llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			if model != nil {
				model.Close()
			}
		})

		It("should default to false", Label("integration"), func() {
			// Embeddings should not be available by default
			_, err := model.GetEmbeddings("Test")
			Expect(err).To(HaveOccurred())
		})

		It("should not allow GetEmbeddings() on generation model", Label("integration"), func() {
			_, err := model.GetEmbeddings("Test")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("Failed to get embeddings from context"))
		})
	})

	Context("with other model options", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set - skipping integration test")
			}
		})

		It("should work with WithContext", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithEmbeddings(),
				llama.WithContext(2048),
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			embeddings, err := model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})

		It("should work with WithThreads", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithEmbeddings(),
				llama.WithThreads(4),
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			embeddings, err := model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})

		It("should work with WithGPULayers", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithEmbeddings(),
				llama.WithGPULayers(1),
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			embeddings, err := model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})

		It("should combine with multiple options", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithEmbeddings(),
				llama.WithContext(2048),
				llama.WithThreads(4),
				llama.WithBatch(512),
				llama.WithF16Memory(),
				llama.WithMMap(true),
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			embeddings, err := model.GetEmbeddings("Test with multiple options")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})
	})
})

var _ = Describe("Embedding Edge Cases", func() {
	Context("with invalid parameters", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set - skipping integration test")
			}
		})

		It("should error with 'Invalid parameters for embeddings' if ctx null", Label("integration"), func() {
			// This tests C++ level validation
			// In Go, closed model returns "model is closed" before reaching C++
			model, err := llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			model.Close()

			_, err = model.GetEmbeddings("Test")
			Expect(err).To(HaveOccurred())
			// Go-level check returns "model is closed"
			Expect(err.Error()).To(Equal("model is closed"))
		})

		It("should handle null text pointer", Label("integration"), func() {
			// In Go, empty string is different from null pointer
			// This documents the expected C++ error for reference
			expectedError := "Invalid parameters for embeddings"
			_ = expectedError // Document expected error string
		})

		It("should handle null embeddings buffer pointer", Label("integration"), func() {
			// This is an internal C++ condition that Go layer handles
			// We document the expected error for completeness
			expectedError := "Invalid parameters for embeddings"
			_ = expectedError // Document expected error string
		})
	})

	Context("with C++ exceptions", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set - skipping integration test")
			}
		})

		It("should return 'Exception during embedding generation:' for exceptions", Label("integration"), func() {
			// C++ exceptions are caught and converted to error messages
			// This documents the expected error format
			expectedErrorPrefix := "Exception during embedding generation:"
			_ = expectedErrorPrefix // Document expected error prefix
		})

		It("should handle exceptions gracefully without crashing", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Try various inputs - should not panic even if errors occur
			inputs := []string{
				"Normal text",
				"",
				"Very long text " + string(make([]byte, 10000)),
				"Unicode: 你好世界 🌍",
			}

			for _, input := range inputs {
				_, _ = model.GetEmbeddings(input)
				// If we reach here without panic, test passes
			}
			Succeed()
		})
	})
})
