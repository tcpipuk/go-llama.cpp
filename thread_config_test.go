package llama_test

import (
	"os"
	"runtime"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	llama "github.com/tcpipuk/llama-go"
)

var _ = Describe("Thread Configuration", Label("thread-config"), func() {
	var modelPath string

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration tests")
		}
	})

	Context("WithThreads", func() {
		It("should respect custom thread count", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(4),
				llama.WithGPULayers(0), // CPU-only to test threads
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Should complete without hanging (threads configured correctly)
			result, err := model.Generate("Hello",
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should use all CPU cores by default", Label("integration"), func() {
			// Default should use runtime.NumCPU() threads
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithGPULayers(0), // CPU-only to test threads
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Hello",
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should handle single thread configuration", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(1),
				llama.WithGPULayers(0), // CPU-only
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Test",
				llama.WithMaxTokens(3),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should handle maximum thread configuration", Label("integration"), func() {
			maxThreads := runtime.NumCPU() * 2
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(maxThreads),
				llama.WithGPULayers(0), // CPU-only
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Test",
				llama.WithMaxTokens(3),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})
	})

	Context("WithThreadsBatch", func() {
		It("should respect custom batch thread count", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(4),
				llama.WithThreadsBatch(8),
				llama.WithGPULayers(0), // CPU-only to test threads
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Should complete without hanging (batch threads configured correctly)
			result, err := model.Generate("Hello",
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should use same as WithThreads by default", Label("integration"), func() {
			// When WithThreadsBatch is 0 (default), should use same as WithThreads
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(6),
				llama.WithThreadsBatch(0), // Explicit default
				llama.WithGPULayers(0),
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Test",
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should allow different batch and prompt thread counts", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(2),
				llama.WithThreadsBatch(8),
				llama.WithGPULayers(0),
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Test",
				llama.WithMaxTokens(10),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})
	})

	Context("thread configuration with GPU", func() {
		It("should work with GPU offloading enabled", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(4),
				llama.WithThreadsBatch(8),
				llama.WithGPULayers(-1), // All layers on GPU
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Hello",
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should work with partial GPU offloading", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(4),
				llama.WithThreadsBatch(6),
				llama.WithGPULayers(10), // Partial offload
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Test",
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})
	})

	Context("edge cases", func() {
		It("should handle batch threads less than prompt threads", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(8),
				llama.WithThreadsBatch(4),
				llama.WithGPULayers(0),
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Test",
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should handle batch threads greater than prompt threads", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(2),
				llama.WithThreadsBatch(16),
				llama.WithGPULayers(0),
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Test",
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should handle equal prompt and batch thread counts", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath,
				llama.WithContext(2048),
				llama.WithThreads(6),
				llama.WithThreadsBatch(6),
				llama.WithGPULayers(0),
			)
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			result, err := model.Generate("Test",
				llama.WithMaxTokens(5),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})
	})
})
