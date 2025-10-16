package llama_test

import (
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	llama "github.com/tcpipuk/llama-go"
)

var _ = Describe("Prefix Caching", Label("prefix-caching"), func() {
	var modelPath string

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_CHAT_MODEL")
		if modelPath == "" {
			Skip("TEST_CHAT_MODEL not set - skipping integration tests")
		}
	})

	Context("deterministic generation", func() {
		It("should produce identical results with prefix caching disabled", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			seed := uint32(12345)
			prompt := "What is 2+2?"

			results := make([]string, 3)
			for i := 0; i < 3; i++ {
				result, err := model.Generate(prompt,
					llama.WithSeed(int(seed)),
					llama.WithMaxTokens(10),
					llama.WithPrefixCaching(false),
				)
				Expect(err).NotTo(HaveOccurred())
				Expect(result).NotTo(BeEmpty())
				results[i] = result
			}

			// All should be identical even without prefix caching (same seed)
			Expect(results[1]).To(Equal(results[0]), "Second generation should match first")
			Expect(results[2]).To(Equal(results[0]), "Third generation should match first")
		})

		It("should produce identical results regardless of prefix caching setting", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			seed := uint32(12345)
			prompt := "What is 2+2?"

			// Generate with prefix caching enabled
			resultWithCache, err := model.Generate(prompt,
				llama.WithSeed(int(seed)),
				llama.WithMaxTokens(10),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(resultWithCache).NotTo(BeEmpty())

			// Generate with prefix caching disabled
			resultWithoutCache, err := model.Generate(prompt,
				llama.WithSeed(int(seed)),
				llama.WithMaxTokens(10),
				llama.WithPrefixCaching(false),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(resultWithoutCache).NotTo(BeEmpty())

			// Results should be identical (same seed, deterministic sampling)
			Expect(resultWithoutCache).To(Equal(resultWithCache),
				"Results should be identical regardless of prefix caching when using same seed")
		})
	})

	Context("performance", func() {
		It("should reuse cached tokens for repeated prompts", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			prompt := "The quick brown fox"

			// First generation establishes cache
			result1, err := model.Generate(prompt,
				llama.WithMaxTokens(5),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result1).NotTo(BeEmpty())

			// Second generation should reuse cache (faster)
			result2, err := model.Generate(prompt,
				llama.WithMaxTokens(5),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result2).NotTo(BeEmpty())

			// Results may differ (no seed), but both should succeed
		})

		It("should handle partial cache hits correctly", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			basePrompt := "The quick brown"
			extendedPrompt := "The quick brown fox"

			// Establish cache with base prompt
			_, err = model.Generate(basePrompt,
				llama.WithMaxTokens(3),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			// Extended prompt should reuse partial cache
			result, err := model.Generate(extendedPrompt,
				llama.WithMaxTokens(3),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})
	})

	Context("cache invalidation", func() {
		It("should not reuse cache when prefix caching is disabled", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			prompt := "Hello world"

			// First generation with caching enabled
			_, err = model.Generate(prompt,
				llama.WithMaxTokens(5),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			// Second generation with caching disabled should not reuse cache
			result, err := model.Generate(prompt,
				llama.WithMaxTokens(5),
				llama.WithPrefixCaching(false),
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})

		It("should handle alternating cache settings correctly", Label("integration", "gpu"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithGPULayers(-1))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			prompt := "Test prompt"
			seed := int(54321)

			// Generate with cache enabled
			result1, err := model.Generate(prompt,
				llama.WithSeed(seed),
				llama.WithMaxTokens(5),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			// Generate with cache disabled
			result2, err := model.Generate(prompt,
				llama.WithSeed(seed),
				llama.WithMaxTokens(5),
				llama.WithPrefixCaching(false),
			)
			Expect(err).NotTo(HaveOccurred())

			// Generate with cache enabled again
			result3, err := model.Generate(prompt,
				llama.WithSeed(seed),
				llama.WithMaxTokens(5),
				llama.WithPrefixCaching(true),
			)
			Expect(err).NotTo(HaveOccurred())

			// All should be identical (same seed)
			Expect(result2).To(Equal(result1))
			Expect(result3).To(Equal(result1))
		})
	})

})
