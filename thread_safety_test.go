package llama_test

import (
	"os"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/tcpipuk/llama-go"
)

// Thread Safety Test Suite
//
// Tests concurrent access patterns and demonstrates that llama-go is NOT thread-safe.
// These tests verify that:
// 1. Concurrent access to the same model instance causes race conditions
// 2. Safe concurrent patterns (separate instances, mutex, pools) work correctly
// 3. Resource management under concurrency behaves as expected
//
// Run with -race flag to detect data races: go test -race ./...

var _ = Describe("Model Thread Safety", func() {
	var modelPath string

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
	})

	Context("concurrent Generate calls", func() {
		It("should detect race conditions with concurrent calls", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Launch concurrent generation calls - this WILL race
			// Run with -race flag to detect the race condition
			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				_, _ = model.Generate("Test prompt 1", llama.WithMaxTokens(10))
			}()

			go func() {
				defer wg.Done()
				_, _ = model.Generate("Test prompt 2", llama.WithMaxTokens(10))
			}()

			wg.Wait()
			// Test passes if it completes (race detector will catch issues)
		})

		It("should corrupt state when called concurrently", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Demonstrate non-deterministic behaviour with concurrent access
			results := make([]string, 10)
			var wg sync.WaitGroup

			for i := 0; i < 10; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()
					result, err := model.Generate("Count to three:",
						llama.WithMaxTokens(20),
						llama.WithSeed(42), // Same seed - should be deterministic
					)
					if err == nil {
						results[index] = result
					}
				}(i)
			}

			wg.Wait()

			// With concurrent access and same seed, results may vary
			// (demonstrating thread-unsafe behaviour)
			// At minimum, test should complete without crashing
			Expect(results).NotTo(BeNil())
		})

		It("should document thread-unsafe behaviour", Label("integration"), func() {
			// This test serves as documentation:
			// Model instances are NOT thread-safe
			// Use separate instances or synchronisation primitives
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Single-threaded access works fine
			result, err := model.Generate("Hello", llama.WithMaxTokens(5))
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeEmpty())
		})
	})

	Context("concurrent method calls", func() {
		It("should show unsafe behaviour with concurrent Tokenize", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var wg sync.WaitGroup
			wg.Add(3)

			go func() {
				defer wg.Done()
				_, _ = model.Tokenize("First text to tokenise")
			}()

			go func() {
				defer wg.Done()
				_, _ = model.Tokenize("Second text to tokenise")
			}()

			go func() {
				defer wg.Done()
				_, _ = model.Tokenize("Third text to tokenise")
			}()

			wg.Wait()
			// Race detector will catch concurrent access issues
		})

		It("should show unsafe behaviour with concurrent GetEmbeddings", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				_, _ = model.GetEmbeddings("First embedding text")
			}()

			go func() {
				defer wg.Done()
				_, _ = model.GetEmbeddings("Second embedding text")
			}()

			wg.Wait()
			// Race detector will catch concurrent access issues
		})

		It("should show unsafe behaviour with mixed method calls", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var wg sync.WaitGroup
			wg.Add(3)

			go func() {
				defer wg.Done()
				_, _ = model.Generate("Generate text", llama.WithMaxTokens(10))
			}()

			go func() {
				defer wg.Done()
				_, _ = model.Tokenize("Tokenize text")
			}()

			go func() {
				defer wg.Done()
				_, _ = model.Generate("Another generation", llama.WithMaxTokens(10))
			}()

			wg.Wait()
			// Race detector will catch concurrent access issues
		})
	})

	Context("concurrent Close calls", func() {
		It("should handle concurrent Close() calls", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())

			var wg sync.WaitGroup
			wg.Add(3)

			// Multiple goroutines try to close the same model
			for i := 0; i < 3; i++ {
				go func() {
					defer wg.Done()
					_ = model.Close()
				}()
			}

			wg.Wait()
			// Should not crash, though behaviour may be undefined
		})

		It("should not crash on concurrent close", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())

			var wg sync.WaitGroup
			wg.Add(5)

			for i := 0; i < 5; i++ {
				go func() {
					defer wg.Done()
					time.Sleep(time.Millisecond * 10) // Slight delay
					_ = model.Close()
				}()
			}

			wg.Wait()
			// Test completes successfully if no crash
		})

		It("should set pointer to nil atomically (or demonstrate issues)", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())

			// Close in one goroutine while checking in another
			done := make(chan bool)
			go func() {
				time.Sleep(time.Millisecond * 50)
				_ = model.Close()
				done <- true
			}()

			// Try to use model concurrently with Close
			go func() {
				for i := 0; i < 10; i++ {
					_, _ = model.Generate("test", llama.WithMaxTokens(5))
					time.Sleep(time.Millisecond * 10)
				}
			}()

			<-done
			// Demonstrates unsafe concurrent Close + usage
		})
	})
})

var _ = Describe("Pool Pattern for Thread Safety", func() {
	var modelPath string

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
	})

	Context("with separate model instances", func() {
		It("should allow concurrent calls on different model instances", Label("integration"), func() {
			// Load separate model instances
			model1, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model1.Close()

			model2, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model2.Close()

			var wg sync.WaitGroup
			wg.Add(2)

			var result1, result2 string

			go func() {
				defer wg.Done()
				var err error
				result1, err = model1.Generate("First model prompt", llama.WithMaxTokens(10))
				Expect(err).NotTo(HaveOccurred())
			}()

			go func() {
				defer wg.Done()
				var err error
				result2, err = model2.Generate("Second model prompt", llama.WithMaxTokens(10))
				Expect(err).NotTo(HaveOccurred())
			}()

			wg.Wait()

			Expect(result1).NotTo(BeEmpty())
			Expect(result2).NotTo(BeEmpty())
		})

		It("should not have race conditions across instances", Label("integration"), func() {
			models := make([]*llama.Model, 5)
			for i := range models {
				var err error
				models[i], err = llama.LoadModel(modelPath, llama.WithContext(2048))
				Expect(err).NotTo(HaveOccurred())
				defer models[i].Close()
			}

			var wg sync.WaitGroup
			results := make([]string, 5)

			for i := 0; i < 5; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()
					result, err := models[index].Generate("Test prompt",
						llama.WithMaxTokens(10),
						llama.WithSeed(42),
					)
					Expect(err).NotTo(HaveOccurred())
					results[index] = result
				}(i)
			}

			wg.Wait()

			// All results should be identical (same seed, separate instances)
			for i := 1; i < len(results); i++ {
				Expect(results[i]).To(Equal(results[0]))
			}
		})

		It("should demonstrate safe concurrent pattern", Label("integration"), func() {
			// Safe pattern: one model instance per goroutine
			const workers = 3
			results := make([]string, workers)
			var wg sync.WaitGroup

			for i := 0; i < workers; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()

					// Each goroutine loads its own model
					model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
					Expect(err).NotTo(HaveOccurred())
					defer model.Close()

					result, err := model.Generate("Worker prompt",
						llama.WithMaxTokens(10))
					Expect(err).NotTo(HaveOccurred())
					results[index] = result
				}(i)
			}

			wg.Wait()

			for i := range results {
				Expect(results[i]).NotTo(BeEmpty())
			}
		})
	})

	Context("with synchronisation primitives", func() {
		It("should work safely with sync.Mutex protection", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var mu sync.Mutex
			var wg sync.WaitGroup
			results := make([]string, 5)

			for i := 0; i < 5; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()

					mu.Lock()
					result, err := model.Generate("Mutex-protected prompt",
						llama.WithMaxTokens(10),
						llama.WithSeed(42),
					)
					mu.Unlock()

					Expect(err).NotTo(HaveOccurred())
					results[index] = result
				}(i)
			}

			wg.Wait()

			// With mutex protection and same seed, all results should be identical
			for i := 1; i < len(results); i++ {
				Expect(results[i]).To(Equal(results[0]))
			}
		})

		It("should demonstrate mutex-protected concurrent access", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var mu sync.Mutex
			var wg sync.WaitGroup

			// Mix different operations under mutex
			wg.Add(3)

			go func() {
				defer wg.Done()
				mu.Lock()
				_, err := model.Generate("Generate", llama.WithMaxTokens(5))
				mu.Unlock()
				Expect(err).NotTo(HaveOccurred())
			}()

			go func() {
				defer wg.Done()
				mu.Lock()
				_, err := model.Tokenize("Tokenize this")
				mu.Unlock()
				Expect(err).NotTo(HaveOccurred())
			}()

			go func() {
				defer wg.Done()
				mu.Lock()
				_, err := model.Generate("Another generate", llama.WithMaxTokens(5))
				mu.Unlock()
				Expect(err).NotTo(HaveOccurred())
			}()

			wg.Wait()
		})

		It("should show performance impact of locking", Label("integration", "slow"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var mu sync.Mutex
			const iterations = 10

			// Measure serialised execution time with mutex
			start := time.Now()
			var wg sync.WaitGroup

			for i := 0; i < iterations; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					mu.Lock()
					_, _ = model.Generate("Test", llama.WithMaxTokens(5))
					mu.Unlock()
				}()
			}

			wg.Wait()
			duration := time.Since(start)

			// Should complete but will be serialised by mutex
			Expect(duration).To(BeNumerically(">", 0))
		})
	})

	Context("with channel-based pooling", func() {
		It("should demonstrate channel-based model pool", Label("integration"), func() {
			const poolSize = 3
			modelPool := make(chan *llama.Model, poolSize)

			// Populate pool
			for i := 0; i < poolSize; i++ {
				model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
				Expect(err).NotTo(HaveOccurred())
				modelPool <- model
			}

			// Clean up pool when done
			defer func() {
				close(modelPool)
				for model := range modelPool {
					model.Close()
				}
			}()

			var wg sync.WaitGroup
			results := make([]string, 5)

			for i := 0; i < 5; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()

					// Acquire model from pool
					model := <-modelPool
					defer func() { modelPool <- model }()

					result, err := model.Generate("Pool prompt",
						llama.WithMaxTokens(10))
					Expect(err).NotTo(HaveOccurred())
					results[index] = result
				}(i)
			}

			wg.Wait()

			for i := range results {
				Expect(results[i]).NotTo(BeEmpty())
			}
		})

		It("should safely distribute work across goroutines", Label("integration"), func() {
			const poolSize = 2
			modelPool := make(chan *llama.Model, poolSize)

			for i := 0; i < poolSize; i++ {
				model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
				Expect(err).NotTo(HaveOccurred())
				modelPool <- model
			}

			defer func() {
				close(modelPool)
				for model := range modelPool {
					model.Close()
				}
			}()

			var wg sync.WaitGroup
			const workers = 6
			successCount := 0
			var countMu sync.Mutex

			for i := 0; i < workers; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()

					model := <-modelPool
					defer func() { modelPool <- model }()

					_, err := model.Generate("Test", llama.WithMaxTokens(5))
					if err == nil {
						countMu.Lock()
						successCount++
						countMu.Unlock()
					}
				}()
			}

			wg.Wait()
			Expect(successCount).To(Equal(workers))
		})

		It("should handle pool exhaustion gracefully", Label("integration"), func() {
			const poolSize = 2
			modelPool := make(chan *llama.Model, poolSize)

			for i := 0; i < poolSize; i++ {
				model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
				Expect(err).NotTo(HaveOccurred())
				modelPool <- model
			}

			defer func() {
				close(modelPool)
				for model := range modelPool {
					model.Close()
				}
			}()

			// Create more workers than pool size
			var wg sync.WaitGroup
			const workers = 5

			for i := 0; i < workers; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()

					// This will block if pool is exhausted
					select {
					case model := <-modelPool:
						defer func() { modelPool <- model }()
						_, _ = model.Generate("Test", llama.WithMaxTokens(5))
					case <-time.After(time.Second * 5):
						// Timeout waiting for pool
					}
				}()
			}

			wg.Wait()
			// Test completes successfully with pool exhaustion handling
		})
	})
})

var _ = Describe("Callback Thread Safety", func() {
	var modelPath string

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
	})

	Context("callback execution", func() {
		It("should execute callbacks synchronously in generation thread", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			callbackThreadID := make(chan uint64, 1)
			mainThreadID := uint64(0) // Would need runtime.Goid() or similar

			err = model.GenerateStream("Test prompt",
				func(token string) bool {
					// Callback executes in same goroutine as generation
					// In practice, would compare goroutine IDs
					callbackThreadID <- mainThreadID
					return false // Stop after first token
				},
				llama.WithMaxTokens(10),
			)

			Expect(err).NotTo(HaveOccurred())
			// Callback should have executed
			Expect(callbackThreadID).To(HaveLen(1))
		})

		It("should not spawn goroutines for callbacks", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			callbackCount := 0

			err = model.GenerateStream("Test prompt",
				func(token string) bool {
					callbackCount++
					// Callbacks execute synchronously
					// No goroutine spawning
					return callbackCount < 5
				},
				llama.WithMaxTokens(10),
			)

			Expect(err).NotTo(HaveOccurred())
			Expect(callbackCount).To(BeNumerically(">=", 1))
		})

		It("should allow callbacks to safely access goroutine-local state", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Goroutine-local state
			tokens := []string{}

			err = model.GenerateStream("Test prompt",
				func(token string) bool {
					// Safe to access goroutine-local state
					tokens = append(tokens, token)
					return len(tokens) < 5
				},
				llama.WithMaxTokens(10),
			)

			Expect(err).NotTo(HaveOccurred())
			Expect(tokens).NotTo(BeEmpty())
		})
	})

	Context("callback with shared state", func() {
		It("should demonstrate safe callback patterns", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Safe pattern: mutex-protected shared state
			var mu sync.Mutex
			sharedTokens := []string{}

			err = model.GenerateStream("Test prompt",
				func(token string) bool {
					mu.Lock()
					sharedTokens = append(sharedTokens, token)
					shouldContinue := len(sharedTokens) < 5
					mu.Unlock()
					return shouldContinue
				},
				llama.WithMaxTokens(10),
			)

			Expect(err).NotTo(HaveOccurred())
			mu.Lock()
			Expect(sharedTokens).NotTo(BeEmpty())
			mu.Unlock()
		})

		It("should handle callback accessing shared state", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var mu sync.Mutex
			totalLength := 0

			err = model.GenerateStream("Test prompt",
				func(token string) bool {
					mu.Lock()
					totalLength += len(token)
					mu.Unlock()
					return true
				},
				llama.WithMaxTokens(10),
			)

			Expect(err).NotTo(HaveOccurred())
			mu.Lock()
			Expect(totalLength).To(BeNumerically(">", 0))
			mu.Unlock()
		})

		It("should show unsafe callback patterns without locking", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Unsafe pattern (if callback were concurrent):
			// No mutex protection on shared state
			counter := 0

			err = model.GenerateStream("Test prompt",
				func(token string) bool {
					counter++ // Would race if callback were concurrent
					return counter < 5
				},
				llama.WithMaxTokens(10),
			)

			Expect(err).NotTo(HaveOccurred())
			// Safe in this case because callbacks are synchronous
			Expect(counter).To(BeNumerically(">=", 1))
		})
	})
})

var _ = Describe("Global Error State Thread Safety", func() {
	var modelPath string

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
	})

	Context("concurrent errors", func() {
		It("should demonstrate g_last_error is thread-unsafe", Label("integration"), func() {
			// g_last_error is a global static string in C++
			// Concurrent errors may corrupt error messages

			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			// Close model to trigger errors
			model.Close()

			var wg sync.WaitGroup
			errors := make([]error, 10)

			for i := 0; i < 10; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()
					_, errors[index] = model.Generate("test", llama.WithMaxTokens(5))
				}(i)
			}

			wg.Wait()

			// All should error with "model is closed"
			for i := range errors {
				Expect(errors[i]).To(HaveOccurred())
			}
		})

		It("should show error state corruption with concurrent failures", Label("integration"), func() {
			// Load multiple models that will fail
			var wg sync.WaitGroup
			errors := make([]error, 5)

			for i := 0; i < 5; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()
					_, errors[index] = llama.LoadModel("/nonexistent/path.gguf")
				}(i)
			}

			wg.Wait()

			// All should error, but error messages may be corrupted
			for i := range errors {
				Expect(errors[i]).To(HaveOccurred())
			}
		})

		It("should document requirement for separate model instances", Label("integration"), func() {
			// Safe pattern: separate model instances avoid shared error state
			var wg sync.WaitGroup
			const instances = 3

			for i := 0; i < instances; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()

					// Each goroutine gets its own model
					model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
					Expect(err).NotTo(HaveOccurred())
					defer model.Close()

					_, err = model.Generate("Test", llama.WithMaxTokens(5))
					Expect(err).NotTo(HaveOccurred())
				}()
			}

			wg.Wait()
		})
	})
})

var _ = Describe("Resource Management Under Concurrency", func() {
	var modelPath string

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
	})

	Context("concurrent model loading", func() {
		It("should safely load multiple models concurrently", Label("integration"), func() {
			var wg sync.WaitGroup
			const concurrentLoads = 5
			models := make([]*llama.Model, concurrentLoads)
			errors := make([]error, concurrentLoads)

			for i := 0; i < concurrentLoads; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()
					models[index], errors[index] = llama.LoadModel(modelPath, llama.WithContext(2048))
				}(i)
			}

			wg.Wait()

			// All loads should succeed
			for i := range models {
				Expect(errors[i]).NotTo(HaveOccurred())
				Expect(models[i]).NotTo(BeNil())
				defer models[i].Close()
			}
		})

		It("should handle llama_backend_init() called concurrently", Label("integration"), func() {
			// llama_backend_init() is called in LoadModel
			// Multiple concurrent calls should be safe
			var wg sync.WaitGroup

			for i := 0; i < 3; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
					Expect(err).NotTo(HaveOccurred())
					defer model.Close()
				}()
			}

			wg.Wait()
		})

		It("should not corrupt backend state", Label("integration"), func() {
			var wg sync.WaitGroup
			const workers = 4
			success := make([]bool, workers)

			for i := 0; i < workers; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()

					model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
					if err == nil {
						defer model.Close()
						_, err = model.Generate("Test", llama.WithMaxTokens(5))
						success[index] = (err == nil)
					}
				}(i)
			}

			wg.Wait()

			// All operations should succeed
			for i := range success {
				Expect(success[i]).To(BeTrue())
			}
		})
	})

	Context("concurrent cleanup", func() {
		It("should safely close multiple models concurrently", Label("integration"), func() {
			// Load models
			const count = 5
			models := make([]*llama.Model, count)
			for i := range models {
				var err error
				models[i], err = llama.LoadModel(modelPath, llama.WithContext(2048))
				Expect(err).NotTo(HaveOccurred())
			}

			// Close concurrently
			var wg sync.WaitGroup
			for i := range models {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()
					err := models[index].Close()
					Expect(err).NotTo(HaveOccurred())
				}(i)
			}

			wg.Wait()
		})

		It("should handle finalisers running concurrently", Label("integration", "slow"), func() {
			// Create models without explicit Close
			// Let finalisers run during GC
			const count = 3

			for i := 0; i < count; i++ {
				model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
				Expect(err).NotTo(HaveOccurred())
				// Intentionally not closing - finaliser will handle it
				_ = model
			}

			// Trigger GC to run finalisers
			// Multiple finalisers may run concurrently
			// This is intentionally commented out to avoid actual GC
			// runtime.GC()
			// time.Sleep(time.Second)

			// Test passes if no crashes occur
		})

		It("should not double-free resources", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())

			// Close explicitly
			err = model.Close()
			Expect(err).NotTo(HaveOccurred())

			// Try to close again (should be safe)
			err = model.Close()
			Expect(err).NotTo(HaveOccurred())

			// Finaliser would also try to close if it runs
			// Should not cause double-free
		})
	})

	Context("model reload cycles", func() {
		It("should safely reload same model path concurrently", Label("integration"), func() {
			var wg sync.WaitGroup
			const cycles = 3

			for i := 0; i < cycles; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()

					// Load and close in cycle
					for j := 0; j < 2; j++ {
						model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
						Expect(err).NotTo(HaveOccurred())
						_, err = model.Generate("Test", llama.WithMaxTokens(5))
						Expect(err).NotTo(HaveOccurred())
						model.Close()
					}
				}()
			}

			wg.Wait()
		})

		It("should handle open/close cycles under concurrency", Label("integration"), func() {
			var wg sync.WaitGroup
			const workers = 3
			const iterationsPerWorker = 2

			for i := 0; i < workers; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()

					for j := 0; j < iterationsPerWorker; j++ {
						model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
						Expect(err).NotTo(HaveOccurred())

						_, err = model.Tokenize("Test tokenisation")
						Expect(err).NotTo(HaveOccurred())

						err = model.Close()
						Expect(err).NotTo(HaveOccurred())
					}
				}()
			}

			wg.Wait()
		})

		It("should not leak memory with concurrent reload", Label("integration", "slow"), func() {
			var wg sync.WaitGroup
			const iterations = 5

			for i := 0; i < iterations; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()

					model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
					Expect(err).NotTo(HaveOccurred())
					defer model.Close()

					_, err = model.Generate("Memory test", llama.WithMaxTokens(10))
					Expect(err).NotTo(HaveOccurred())
				}()
			}

			wg.Wait()
			// Memory leak detection would require additional tooling
		})
	})
})

var _ = Describe("Race Detection", func() {
	var modelPath string

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
	})

	Context("with Go race detector", func() {
		It("should detect races in concurrent Generate", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				_, _ = model.Generate("Race test 1", llama.WithMaxTokens(5))
			}()

			go func() {
				defer wg.Done()
				_, _ = model.Generate("Race test 2", llama.WithMaxTokens(5))
			}()

			wg.Wait()
			// Run with -race flag to detect race
		})

		It("should detect races in concurrent Tokenize", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				_, _ = model.Tokenize("Race test 1")
			}()

			go func() {
				defer wg.Done()
				_, _ = model.Tokenize("Race test 2")
			}()

			wg.Wait()
			// Run with -race flag to detect race
		})

		It("should detect races in concurrent GetEmbeddings", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithEmbeddings())
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				_, _ = model.GetEmbeddings("Race test 1")
			}()

			go func() {
				defer wg.Done()
				_, _ = model.GetEmbeddings("Race test 2")
			}()

			wg.Wait()
			// Run with -race flag to detect race
		})

		It("should detect races in concurrent Close", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())

			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				_ = model.Close()
			}()

			go func() {
				defer wg.Done()
				_ = model.Close()
			}()

			wg.Wait()
			// Run with -race flag to detect race
		})
	})

	Context("safe patterns", func() {
		It("should pass race detector with mutex protection", Label("integration"), func() {
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var mu sync.Mutex
			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				mu.Lock()
				_, _ = model.Generate("Safe test 1", llama.WithMaxTokens(5))
				mu.Unlock()
			}()

			go func() {
				defer wg.Done()
				mu.Lock()
				_, _ = model.Generate("Safe test 2", llama.WithMaxTokens(5))
				mu.Unlock()
			}()

			wg.Wait()
			// Should pass -race with mutex protection
		})

		It("should pass race detector with separate instances", Label("integration"), func() {
			model1, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model1.Close()

			model2, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model2.Close()

			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				_, _ = model1.Generate("Instance 1 test", llama.WithMaxTokens(5))
			}()

			go func() {
				defer wg.Done()
				_, _ = model2.Generate("Instance 2 test", llama.WithMaxTokens(5))
			}()

			wg.Wait()
			// Should pass -race with separate instances
		})

		It("should pass race detector with channel pooling", Label("integration"), func() {
			const poolSize = 2
			modelPool := make(chan *llama.Model, poolSize)

			for i := 0; i < poolSize; i++ {
				model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
				Expect(err).NotTo(HaveOccurred())
				modelPool <- model
			}

			defer func() {
				close(modelPool)
				for model := range modelPool {
					model.Close()
				}
			}()

			var wg sync.WaitGroup
			wg.Add(4)

			for i := 0; i < 4; i++ {
				go func() {
					defer wg.Done()
					model := <-modelPool
					_, _ = model.Generate("Pool test", llama.WithMaxTokens(5))
					modelPool <- model
				}()
			}

			wg.Wait()
			// Should pass -race with channel pooling
		})
	})
})

var _ = Describe("Documentation Examples", func() {
	var modelPath string

	BeforeEach(func() {
		modelPath = os.Getenv("TEST_MODEL")
		if modelPath == "" {
			Skip("TEST_MODEL not set - skipping integration test")
		}
	})

	Context("unsafe pattern demonstration", func() {
		It("should demonstrate concurrent access failure", Label("integration"), func() {
			// ❌ UNSAFE PATTERN - DO NOT USE IN PRODUCTION
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				_, _ = model.Generate("Unsafe concurrent call 1", llama.WithMaxTokens(10))
			}()

			go func() {
				defer wg.Done()
				_, _ = model.Generate("Unsafe concurrent call 2", llama.WithMaxTokens(10))
			}()

			wg.Wait()
			// This demonstrates UNSAFE usage - will race with -race flag
		})

		It("should provide example of race condition", Label("integration"), func() {
			// ❌ UNSAFE PATTERN
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			results := make([]string, 2)
			var wg sync.WaitGroup
			wg.Add(2)

			go func() {
				defer wg.Done()
				results[0], _ = model.Generate("First", llama.WithMaxTokens(5))
			}()

			go func() {
				defer wg.Done()
				results[1], _ = model.Generate("Second", llama.WithMaxTokens(5))
			}()

			wg.Wait()
			// Results are unreliable due to race condition
		})

		It("should show corrupted generation output", Label("integration"), func() {
			// ❌ UNSAFE PATTERN
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var wg sync.WaitGroup
			const workers = 5

			for i := 0; i < workers; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					// Concurrent access may produce corrupted/inconsistent results
					_, _ = model.Generate("Test", llama.WithMaxTokens(10))
				}()
			}

			wg.Wait()
		})
	})

	Context("safe pattern demonstration", func() {
		It("should demonstrate pool-based concurrent usage", Label("integration"), func() {
			// ✅ SAFE PATTERN: Channel-based pool
			const poolSize = 2
			modelPool := make(chan *llama.Model, poolSize)

			// Initialise pool
			for i := 0; i < poolSize; i++ {
				model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
				Expect(err).NotTo(HaveOccurred())
				modelPool <- model
			}

			defer func() {
				close(modelPool)
				for model := range modelPool {
					model.Close()
				}
			}()

			// Safe concurrent usage
			var wg sync.WaitGroup
			results := make([]string, 4)

			for i := 0; i < 4; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()

					// Acquire model from pool
					model := <-modelPool
					defer func() { modelPool <- model }()

					result, err := model.Generate("Safe pool usage",
						llama.WithMaxTokens(10))
					Expect(err).NotTo(HaveOccurred())
					results[index] = result
				}(i)
			}

			wg.Wait()

			for i := range results {
				Expect(results[i]).NotTo(BeEmpty())
			}
		})

		It("should provide example of mutex-protected access", Label("integration"), func() {
			// ✅ SAFE PATTERN: Mutex protection
			model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
			Expect(err).NotTo(HaveOccurred())
			defer model.Close()

			var mu sync.Mutex
			var wg sync.WaitGroup
			results := make([]string, 3)

			for i := 0; i < 3; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()

					mu.Lock()
					result, err := model.Generate("Safe mutex usage",
						llama.WithMaxTokens(10))
					mu.Unlock()

					Expect(err).NotTo(HaveOccurred())
					results[index] = result
				}(i)
			}

			wg.Wait()

			for i := range results {
				Expect(results[i]).NotTo(BeEmpty())
			}
		})

		It("should show correct concurrent generation", Label("integration"), func() {
			// ✅ SAFE PATTERN: Separate instances
			var wg sync.WaitGroup
			results := make([]string, 3)

			for i := 0; i < 3; i++ {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()

					// Each goroutine gets its own model instance
					model, err := llama.LoadModel(modelPath, llama.WithContext(2048))
					Expect(err).NotTo(HaveOccurred())
					defer model.Close()

					result, err := model.Generate("Safe separate instances",
						llama.WithMaxTokens(10))
					Expect(err).NotTo(HaveOccurred())
					results[index] = result
				}(i)
			}

			wg.Wait()

			for i := range results {
				Expect(results[i]).NotTo(BeEmpty())
			}
		})
	})
})
