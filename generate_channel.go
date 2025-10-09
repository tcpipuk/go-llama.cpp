package llama

import (
	gocontext "context"
)

// GenerateChannel generates text with channel-based streaming.
//
// This method returns two channels: one for streaming tokens and one for errors.
// Both channels are closed when generation completes, either successfully or due
// to cancellation or error.
//
// The token channel is buffered (256 tokens) to prevent blocking GPU inference
// when the consumer processes tokens slowly. The error channel has a small buffer
// to prevent goroutine leaks if errors occur when nobody is listening.
//
// Cancelling the context immediately stops generation. Any tokens generated before
// cancellation are available in the channel.
//
// Example:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
//	defer cancel()
//
//	tokenCh, errCh := model.GenerateChannel(ctx, "Once upon a time",
//	    llama.WithMaxTokens(100))
//
//	for {
//	    select {
//	    case token, ok := <-tokenCh:
//	        if !ok {
//	            return // Generation complete
//	        }
//	        fmt.Print(token)
//	    case err := <-errCh:
//	        if err != nil {
//	            log.Fatal(err)
//	        }
//	    case <-ctx.Done():
//	        return
//	    }
//	}
func (m *Model) GenerateChannel(ctx gocontext.Context, prompt string, opts ...GenerateOption) (<-chan string, <-chan error) {
	tokenCh := make(chan string, 256) // Buffered for performance
	errCh := make(chan error, 1)      // Small buffer to prevent blocking

	go func() {
		defer close(tokenCh) // ALWAYS close on exit
		defer close(errCh)

		// Use existing GenerateStream with callback wrapper
		err := m.GenerateStream(prompt, func(token string) bool {
			select {
			case tokenCh <- token:
				return true // Continue generation
			case <-ctx.Done():
				return false // Stop generation on context cancel
			}
		}, opts...)

		if err != nil {
			select {
			case errCh <- err:
			default: // Don't block if nobody listening
			}
		}
	}()

	return tokenCh, errCh
}

// GenerateWithDraftChannel performs speculative generation with channel-based streaming.
//
// This method combines speculative decoding (using a draft model for faster generation)
// with channel-based streaming. It returns two channels: one for verified tokens and
// one for errors.
//
// Both channels are closed when generation completes. The token channel is buffered
// to prevent blocking GPU inference during speculative decoding.
//
// Cancelling the context immediately stops generation.
//
// Example:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
//	defer cancel()
//
//	target, _ := llama.LoadModel("large-model.gguf")
//	draft, _ := llama.LoadModel("small-model.gguf")
//
//	tokenCh, errCh := target.GenerateWithDraftChannel(ctx, "Write a story", draft,
//	    llama.WithDraftTokens(8),
//	    llama.WithMaxTokens(256))
//
//	for {
//	    select {
//	    case token, ok := <-tokenCh:
//	        if !ok {
//	            return // Generation complete
//	        }
//	        fmt.Print(token)
//	    case err := <-errCh:
//	        if err != nil {
//	            log.Fatal(err)
//	        }
//	    case <-ctx.Done():
//	        return
//	    }
//	}
func (m *Model) GenerateWithDraftChannel(ctx gocontext.Context, prompt string, draft *Model, opts ...GenerateOption) (<-chan string, <-chan error) {
	tokenCh := make(chan string, 256) // Buffered for performance
	errCh := make(chan error, 1)      // Small buffer to prevent blocking

	go func() {
		defer close(tokenCh) // ALWAYS close on exit
		defer close(errCh)

		// Use existing GenerateWithDraftStream with callback wrapper
		err := m.GenerateWithDraftStream(prompt, draft, func(token string) bool {
			select {
			case tokenCh <- token:
				return true // Continue generation
			case <-ctx.Done():
				return false // Stop generation on context cancel
			}
		}, opts...)

		if err != nil {
			select {
			case errCh <- err:
			default: // Don't block if nobody listening
			}
		}
	}()

	return tokenCh, errCh
}
