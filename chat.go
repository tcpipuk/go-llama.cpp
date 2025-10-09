package llama

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

import (
	gocontext "context"
	"fmt"
	"strings"
	"unsafe"
)

// Chat performs non-streaming chat completion.
//
// This is a convenience method that collects all streaming deltas and returns
// the complete response. For streaming output, use ChatStream instead.
//
// The method builds a prompt from the provided messages using a simple chat
// template format. Future versions will integrate with llama.cpp's native
// chat template system for proper model-specific formatting.
//
// Example:
//
//	messages := []llama.ChatMessage{
//	    {Role: "system", Content: "You are a helpful assistant."},
//	    {Role: "user", Content: "What is the capital of France?"},
//	}
//
//	response, err := model.Chat(context.Background(), messages, llama.ChatOptions{
//	    MaxTokens: llama.Int(100),
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(response.Content)
func (m *Model) Chat(ctx gocontext.Context, messages []ChatMessage, opts ChatOptions) (*ChatResponse, error) {
	// Build prompt from messages using chat template
	prompt, err := formatChatMessages(m, messages, opts)
	if err != nil {
		return nil, err
	}

	// Build generation options from chat options
	// Use user-provided stop words (no defaults - template handles this)
	genOpts := []GenerateOption{
		WithStopWords(opts.StopWords...),
	}

	if opts.MaxTokens != nil {
		genOpts = append(genOpts, WithMaxTokens(*opts.MaxTokens))
	}
	if opts.Temperature != nil {
		genOpts = append(genOpts, WithTemperature(*opts.Temperature))
	}
	if opts.TopP != nil {
		genOpts = append(genOpts, WithTopP(*opts.TopP))
	}
	if opts.TopK != nil {
		genOpts = append(genOpts, WithTopK(*opts.TopK))
	}
	if opts.Seed != nil {
		genOpts = append(genOpts, WithSeed(*opts.Seed))
	}

	// Generate using existing GenerateChannel
	tokenCh, errCh := m.GenerateChannel(ctx, prompt, genOpts...)

	var content strings.Builder

Loop:
	for {
		select {
		case token, ok := <-tokenCh:
			if !ok {
				break Loop
			}
			content.WriteString(token)
		case err := <-errCh:
			if err != nil {
				return nil, err
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	// Parse final output to extract reasoning
	fullOutput := content.String()
	chatFormat := m.getChatFormat()
	parsedContent, reasoning, err := parseReasoning(fullOutput, opts.ReasoningFormat, chatFormat)
	if err != nil {
		// If parsing fails, return content as-is without reasoning extraction
		return &ChatResponse{Content: fullOutput}, nil
	}

	return &ChatResponse{
		Content:          parsedContent,
		ReasoningContent: reasoning,
	}, nil
}

// ChatStream performs streaming chat completion with channels.
//
// Returns two channels: one for streaming deltas and one for errors.
// Both channels are closed when completion finishes. The delta channel
// is buffered to prevent blocking model inference.
//
// For reasoning models, deltas may contain either Content or ReasoningContent
// (or both, though typically one or the other). Clients should handle both
// fields appropriately.
//
// The goroutine managing the stream will exit when:
//   - Generation completes naturally
//   - An error occurs
//   - The context is cancelled
//
// Example:
//
//	messages := []llama.ChatMessage{
//	    {Role: "user", Content: "Write a story"},
//	}
//
//	deltaCh, errCh := model.ChatStream(context.Background(), messages,
//	    llama.ChatOptions{MaxTokens: llama.Int(200)})
//
//	for {
//	    select {
//	    case delta, ok := <-deltaCh:
//	        if !ok {
//	            return
//	        }
//	        if delta.Content != "" {
//	            fmt.Print(delta.Content)
//	        }
//	        if delta.ReasoningContent != "" {
//	            // Handle reasoning output separately
//	            fmt.Print("[thinking: ", delta.ReasoningContent, "]")
//	        }
//	    case err := <-errCh:
//	        if err != nil {
//	            log.Fatal(err)
//	        }
//	    }
//	}
func (m *Model) ChatStream(ctx gocontext.Context, messages []ChatMessage, opts ChatOptions) (<-chan ChatDelta, <-chan error) {
	bufferSize := 256
	if opts.StreamBufferSize > 0 {
		bufferSize = opts.StreamBufferSize
	}

	deltaCh := make(chan ChatDelta, bufferSize)
	errCh := make(chan error, 1)

	go func() {
		defer close(deltaCh)
		defer close(errCh)

		// Build prompt from messages using chat template
		prompt, err := formatChatMessages(m, messages, opts)
		if err != nil {
			select {
			case errCh <- err:
			default:
			}
			return
		}

		// Build generation options from chat options
		// Use user-provided stop words (no defaults - template handles this)
		genOpts := []GenerateOption{
			WithStopWords(opts.StopWords...),
		}

		if opts.MaxTokens != nil {
			genOpts = append(genOpts, WithMaxTokens(*opts.MaxTokens))
		}
		if opts.Temperature != nil {
			genOpts = append(genOpts, WithTemperature(*opts.Temperature))
		}
		if opts.TopP != nil {
			genOpts = append(genOpts, WithTopP(*opts.TopP))
		}
		if opts.TopK != nil {
			genOpts = append(genOpts, WithTopK(*opts.TopK))
		}
		if opts.Seed != nil {
			genOpts = append(genOpts, WithSeed(*opts.Seed))
		}

		// Use existing GenerateChannel
		tokenCh, genErrCh := m.GenerateChannel(ctx, prompt, genOpts...)

		// Get chat format once before loop
		chatFormat := m.getChatFormat()

		// Track accumulated output and previous parsed state for delta computation
		var accumulated strings.Builder
		var prevContent, prevReasoning string

	Loop:
		for {
			select {
			case token, ok := <-tokenCh:
				if !ok {
					break Loop
				}

				// Accumulate token
				accumulated.WriteString(token)

				// Parse accumulated output to extract reasoning
				content, reasoning, err := parseReasoning(accumulated.String(), opts.ReasoningFormat, chatFormat)
				if err != nil {
					// If parsing fails, send token as-is without reasoning extraction
					select {
					case deltaCh <- ChatDelta{Content: token}:
					case <-ctx.Done():
						return
					}
					continue
				}

				// Compute deltas (what's new since last parse)
				contentDelta := content[len(prevContent):]
				reasoningDelta := reasoning[len(prevReasoning):]

				// Send delta if there's new content or reasoning
				if contentDelta != "" || reasoningDelta != "" {
					select {
					case deltaCh <- ChatDelta{
						Content:          contentDelta,
						ReasoningContent: reasoningDelta,
					}:
					case <-ctx.Done():
						return
					}
				}

				// Update previous state
				prevContent = content
				prevReasoning = reasoning

			case err := <-genErrCh:
				if err != nil {
					select {
					case errCh <- err:
					default:
					}
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	return deltaCh, errCh
}

// formatChatMessages applies the model's chat template to messages.
//
// This uses llama.cpp's native chat template system which supports 40+ formats
// including chatml, llama2, llama3, mistral, gemma, phi3, and more. The template
// is read from the model's GGUF metadata or provided via ChatOptions.ChatTemplate.
//
// Returns an error if no template is available (neither in options nor model metadata).
// For raw completion without templates, use Generate() instead of Chat().
func formatChatMessages(model *Model, messages []ChatMessage, opts ChatOptions) (string, error) {
	// Priority: user-provided template > model's GGUF template > error
	template := opts.ChatTemplate
	if template == "" {
		template = model.ChatTemplate()
	}
	if template == "" {
		return "", fmt.Errorf("no chat template available: provide ChatOptions.ChatTemplate or use a model with embedded template (or use Generate() for raw completion)")
	}

	// Apply template using native llama.cpp implementation
	prompt, err := applyChatTemplate(template, messages, true)
	if err != nil {
		return "", fmt.Errorf("failed to apply chat template: %w", err)
	}

	return prompt, nil
}

// parseReasoning extracts reasoning/thinking content from model output.
// Returns content and reasoning_content separately.
func parseReasoning(text string, format ReasoningFormat, chatFormat int) (content, reasoningContent string, err error) {
	if format == ReasoningFormatNone || text == "" {
		return text, "", nil
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cFormat := C.llama_wrapper_reasoning_format(format)
	cChatFormat := C.int(chatFormat)

	// Parse with is_partial=true for streaming
	result := C.llama_wrapper_parse_reasoning(cText, C.bool(true), cFormat, cChatFormat)
	if result == nil {
		return "", "", fmt.Errorf("failed to parse reasoning: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	defer C.llama_wrapper_free_parsed_message(result)

	content = C.GoString(result.content)
	if result.reasoning_content != nil {
		reasoningContent = C.GoString(result.reasoning_content)
	}

	return content, reasoningContent, nil
}

// Int returns a pointer to the given int value.
// This is a convenience helper for setting optional ChatOptions fields.
//
// Example:
//
//	opts := llama.ChatOptions{
//	    MaxTokens: llama.Int(100),  // Instead of &100
//	}
func Int(v int) *int {
	return &v
}

// Float32 returns a pointer to the given float32 value.
// This is a convenience helper for setting optional ChatOptions fields.
//
// Example:
//
//	opts := llama.ChatOptions{
//	    Temperature: llama.Float32(0.7),  // Instead of &0.7
//	}
func Float32(v float32) *float32 {
	return &v
}

// Bool returns a pointer to the given bool value.
// This is a convenience helper for setting optional ChatOptions fields.
//
// Example:
//
//	opts := llama.ChatOptions{
//	    EnableThinking: llama.Bool(true),  // Instead of &true
//	}
func Bool(v bool) *bool {
	return &v
}
