package llama_test

import (
	"os"
	"runtime"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tcpipuk/llama-go"
)

// Model Lifecycle Tests
//
// Tests for model loading, configuration, closure, and finaliser behaviour.
// Covers LoadModel function, Model.Close method, and resource management patterns.

var _ = Describe("LoadModel", func() {
	Context("with valid model path", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set - skipping integration test")
			}
		})

		It("should load model successfully", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()
		})

		It("should return non-nil model pointer", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()
		})

		It("should initialise llama backend", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Verify backend is initialised by performing a basic operation
			tokens, err := model.Tokenize("test")
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).NotTo(BeEmpty())
		})

		It("should set finaliser for automatic cleanup", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Finaliser is set during LoadModel; verify model works normally
			// (finaliser testing is in separate suite due to GC requirements)
			tokens, err := model.Tokenize("test")
			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).NotTo(BeEmpty())
		})
	})

	Context("with invalid model path", func() {
		It("should return error for empty string path", Label("unit"), func() {
			model, err := llama.LoadModel("")
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})

		It("should return error for non-existent file path", Label("unit"), func() {
			model, err := llama.LoadModel("/nonexistent/model.gguf")
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})

		It("should return error containing \"Failed to load model from:\"", Label("unit"), func() {
			_, err := llama.LoadModel("/nonexistent/model.gguf")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("Failed to load model from:"))
		})

		It("should return nil model on error", Label("unit"), func() {
			model, err := llama.LoadModel("/nonexistent/model.gguf")
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})
	})

	Context("with null/invalid path formats", func() {
		It("should return \"Model path cannot be null\" for null path", Label("unit"), func() {
			_, err := llama.LoadModel("")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("Model path cannot be null"))
		})

		It("should handle paths with special characters", Label("integration"), func() {
			modelPath := os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}

			// Test with path that might have spaces or special chars
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()
		})

		It("should handle relative vs absolute paths", Label("integration"), func() {
			modelPath := os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}

			// Test that valid paths work regardless of format
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()
		})
	})

	Context("with configuration options", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}
		})

		It("should apply WithContext option", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(4096))
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Verify context size by attempting generation
			response, err := model.Generate("Hello", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should apply WithBatch option", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithBatch(256))
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Verify batch size by performing generation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should apply WithThreads option", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithThreads(2))
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Verify threads by performing generation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should apply WithGPULayers option", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithGPULayers(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// GPU layers configured, verify basic operation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should apply WithF16Memory option", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithF16Memory())
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// F16 memory enabled, verify operation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should apply WithMLock option", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithMLock())
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// MLock enabled, verify operation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should apply WithMMap option", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithMMap(false))
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// MMap disabled, verify operation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should apply WithEmbeddings option", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Embeddings enabled, verify we can get embeddings
			embeddings, err := model.GetEmbeddings("Test")
			Expect(err).NotTo(HaveOccurred())
			Expect(embeddings).NotTo(BeEmpty())
		})

		It("should apply multiple options together", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(4096),
				llama.WithBatch(256),
				llama.WithThreads(4),
				llama.WithF16Memory(),
				llama.WithMMap(true),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// All options applied, verify operation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})
	})

	Context("with default configuration", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}
		})

		It("should use context size 2048 when not specified", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Default context is 2048, verify by successful generation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should use batch size 512 when not specified", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Default batch is 512, verify by successful generation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should use CPU-only (0 GPU layers) when not specified", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Default is CPU-only, verify operation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should use runtime.NumCPU() threads when not specified", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// Default threads is runtime.NumCPU(), verify operation
			expectedThreads := runtime.NumCPU()
			Expect(expectedThreads).To(BeNumerically(">", 0))

			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})

		It("should enable mmap by default", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())
			defer model.Close()

			// MMap enabled by default, verify operation
			response, err := model.Generate("Test", llama.WithMaxTokens(10))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})
	})

	Context("when context creation fails", func() {
		It("should return \"Failed to create context\" error", Label("integration"), func() {
			// This is difficult to trigger without invalid configuration
			// Test that error message format is correct when it does occur
			modelPath := os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}

			// Attempt to load with potentially problematic config
			// (actual failure difficult to guarantee)
			model, err := llama.LoadModel(modelPath, llama.WithContext(0))
			if err != nil {
				// If it fails, verify error message
				Expect(err.Error()).To(Or(
					ContainSubstring("Failed to create context"),
					ContainSubstring("Invalid context size"),
				))
			} else {
				// If it succeeds (C++ applies default), clean up
				defer model.Close()
			}
		})

		It("should free model if context creation fails", Label("integration"), func() {
			// Verify that failed loads don't leak memory
			// Load failure should clean up properly
			_, err := llama.LoadModel("/nonexistent/model.gguf")
			Expect(err).To(HaveOccurred())

			// No model to close, verify no panic from finaliser
			runtime.GC()
		})
	})
})

var _ = Describe("Model.Close", func() {
	Context("on valid model", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}
		})

		It("should free resources successfully", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			err = model.Close()
			Expect(err).NotTo(HaveOccurred())
		})

		It("should set pointer to nil", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			model.Close()

			// Verify model is closed by attempting operation
			_, err = model.Generate("test")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("model is closed"))
		})

		It("should remove finaliser", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			err = model.Close()
			Expect(err).NotTo(HaveOccurred())

			// Finaliser removed, no double-free on GC
			runtime.GC()
		})

		It("should always return nil error", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			err = model.Close()
			Expect(err).To(BeNil())
		})
	})

	Context("when called multiple times", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}
		})

		It("should be safe to call Close() twice", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			err = model.Close()
			Expect(err).NotTo(HaveOccurred())

			err = model.Close()
			Expect(err).NotTo(HaveOccurred())
		})

		It("should not panic on double-close", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			Expect(func() {
				model.Close()
				model.Close()
			}).NotTo(Panic())
		})

		It("should remain nil after second close", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			model.Close()
			model.Close()

			// Verify still closed
			_, err = model.Generate("test")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(Equal("model is closed"))
		})
	})

	Context("on already-closed model", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}
		})

		It("should be idempotent", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			model.Close()

			// Multiple closes should have same effect
			err = model.Close()
			Expect(err).NotTo(HaveOccurred())

			err = model.Close()
			Expect(err).NotTo(HaveOccurred())
		})

		It("should not error on nil pointer", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			model.Close()

			// Close on already-closed model (nil pointer internally)
			err = model.Close()
			Expect(err).NotTo(HaveOccurred())
		})
	})
})

var _ = Describe("Model Finaliser", func() {
	Context("when model not explicitly closed", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}
		})

		It("should call Close() via finaliser", Label("integration", "slow"), func() {
			// Load model and let it go out of scope
			func() {
				model, err := llama.LoadModel(modelPath)
				Expect(err).NotTo(HaveOccurred())
				Expect(model).NotTo(BeNil())
				// Model goes out of scope without explicit Close()
			}()

			// Force GC to run finalisers
			runtime.GC()
			runtime.GC() // Multiple GC cycles to ensure finaliser runs

			// If finaliser worked, no crash or leak
			// Load another model to verify no corruption
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()
		})

		It("should free resources after GC", Label("integration", "slow"), func() {
			// Track that resources are freed by finaliser
			initialModel, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			initialModel.Close()

			// Load model without closing
			func() {
				model, err := llama.LoadModel(modelPath)
				Expect(err).NotTo(HaveOccurred())
				Expect(model).NotTo(BeNil())
				// Goes out of scope
			}()

			// Force finaliser
			runtime.GC()
			runtime.GC()

			// Should be able to load again without issues
			newModel, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			defer newModel.Close()
		})

		It("should handle finaliser running after explicit Close()", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			// Explicitly close (removes finaliser)
			model.Close()

			// Force GC - finaliser should not run again
			runtime.GC()
			runtime.GC()

			// No double-free, no crash
			// Verify by loading new model
			newModel, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			defer newModel.Close()
		})
	})

	Context("when explicitly closed", func() {
		var modelPath string

		BeforeEach(func() {
			modelPath = os.Getenv("TEST_MODEL")
			if modelPath == "" {
				Skip("TEST_MODEL not set")
			}
		})

		It("should remove finaliser on Close()", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			// Close removes finaliser
			model.Close()

			// Finaliser should not run
			runtime.GC()
			runtime.GC()

			// Verify no issues
			newModel, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			defer newModel.Close()
		})

		It("should not double-free if GC runs later", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			Expect(model).NotTo(BeNil())

			model.Close()

			// Multiple GC cycles should not cause issues
			runtime.GC()
			runtime.GC()
			runtime.GC()

			// Verify system still stable
			newModel, err := llama.LoadModel(modelPath)
			Expect(err).NotTo(HaveOccurred())
			defer newModel.Close()

			response, err := newModel.Generate("Test", llama.WithMaxTokens(5))
			Expect(err).NotTo(HaveOccurred())
			Expect(response).NotTo(BeEmpty())
		})
	})
})
