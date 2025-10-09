package llama

import (
	"fmt"
	"time"
	"unsafe"
)

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

// newContextPool creates a new context pool with min contexts initialised
func newContextPool(model unsafe.Pointer, config modelConfig) (*contextPool, error) {
	pool := &contextPool{
		model:     model,
		config:    config,
		contexts:  make([]*context, 0, config.maxContexts),
		available: make(chan *context, config.maxContexts),
		done:      make(chan struct{}),
	}

	// Convert Go config to C struct for context creation
	var cMainGPU *C.char
	if config.mainGPU != "" {
		cMainGPU = C.CString(config.mainGPU)
		defer C.free(unsafe.Pointer(cMainGPU))
	}

	var cTensorSplit *C.char
	if config.tensorSplit != "" {
		cTensorSplit = C.CString(config.tensorSplit)
		defer C.free(unsafe.Pointer(cTensorSplit))
	}

	var cKVCacheType *C.char
	if config.kvCacheType != "" {
		cKVCacheType = C.CString(config.kvCacheType)
		defer C.free(unsafe.Pointer(cKVCacheType))
	}

	params := C.llama_wrapper_model_params{
		n_ctx:           C.int(config.contextSize),
		n_batch:         C.int(config.batchSize),
		n_gpu_layers:    C.int(config.gpuLayers),
		n_threads:       C.int(config.threads),
		n_threads_batch: C.int(config.threadsBatch),
		f16_memory:      C.bool(config.f16Memory),
		mlock:           C.bool(config.mlock),
		mmap:            C.bool(config.mmap),
		embeddings:      C.bool(config.embeddings),
		main_gpu:        cMainGPU,
		tensor_split:    cTensorSplit,
		kv_cache_type:   cKVCacheType,
	}

	// Create min contexts upfront
	for i := 0; i < config.minContexts; i++ {
		ctxPtr := C.llama_wrapper_context_create(model, params)
		if ctxPtr == nil {
			// Clean up already created contexts
			pool.close()
			return nil, fmt.Errorf("failed to create initial context %d: %s", i, C.GoString(C.llama_wrapper_last_error()))
		}

		ctx := &context{
			ptr:     ctxPtr,
			lastUse: time.Now(),
		}
		pool.contexts = append(pool.contexts, ctx)
		pool.available <- ctx
	}

	// Start cleanup goroutine
	pool.wg.Add(1)
	go pool.cleanup()

	return pool, nil
}

// acquire gets a context from the pool, creating one if needed
func (p *contextPool) acquire() (*context, error) {
	select {
	case ctx := <-p.available:
		// Got an available context
		ctx.mu.Lock()
		return ctx, nil
	case <-p.done:
		return nil, fmt.Errorf("pool is closed")
	default:
		// No available context - try to create one
		p.mu.Lock()
		if len(p.contexts) < p.config.maxContexts {
			// We can create a new context
			var cMainGPU *C.char
			if p.config.mainGPU != "" {
				cMainGPU = C.CString(p.config.mainGPU)
				defer C.free(unsafe.Pointer(cMainGPU))
			}

			var cTensorSplit *C.char
			if p.config.tensorSplit != "" {
				cTensorSplit = C.CString(p.config.tensorSplit)
				defer C.free(unsafe.Pointer(cTensorSplit))
			}

			var cKVCacheType *C.char
			if p.config.kvCacheType != "" {
				cKVCacheType = C.CString(p.config.kvCacheType)
				defer C.free(unsafe.Pointer(cKVCacheType))
			}

			params := C.llama_wrapper_model_params{
				n_ctx:           C.int(p.config.contextSize),
				n_batch:         C.int(p.config.batchSize),
				n_gpu_layers:    C.int(p.config.gpuLayers),
				n_threads:       C.int(p.config.threads),
				n_threads_batch: C.int(p.config.threadsBatch),
				f16_memory:      C.bool(p.config.f16Memory),
				mlock:           C.bool(p.config.mlock),
				mmap:            C.bool(p.config.mmap),
				embeddings:      C.bool(p.config.embeddings),
				main_gpu:        cMainGPU,
				tensor_split:    cTensorSplit,
				kv_cache_type:   cKVCacheType,
			}

			ctxPtr := C.llama_wrapper_context_create(p.model, params)
			if ctxPtr == nil {
				p.mu.Unlock()
				return nil, fmt.Errorf("failed to create context: %s", C.GoString(C.llama_wrapper_last_error()))
			}

			ctx := &context{
				ptr:     ctxPtr,
				lastUse: time.Now(),
			}
			p.contexts = append(p.contexts, ctx)
			p.mu.Unlock()

			ctx.mu.Lock()
			return ctx, nil
		}
		p.mu.Unlock()

		// Max contexts reached - wait for one to become available
		select {
		case ctx := <-p.available:
			ctx.mu.Lock()
			return ctx, nil
		case <-p.done:
			return nil, fmt.Errorf("pool is closed")
		}
	}
}

// release returns a context to the pool
func (p *contextPool) release(ctx *context) {
	ctx.lastUse = time.Now()
	ctx.mu.Unlock()

	select {
	case p.available <- ctx:
		// Context returned to pool
	case <-p.done:
		// Pool is closed, don't return
	}
}

// cleanup runs periodically to destroy idle contexts
func (p *contextPool) cleanup() {
	defer p.wg.Done()

	// Wait for shutdown signal
	// Note: Idle context cleanup is disabled to avoid complexity during testing
	// In production, this could be re-enabled with proper synchronisation
	<-p.done
}

// close shuts down the pool and frees all contexts
func (p *contextPool) close() {
	close(p.done)
	p.wg.Wait()

	p.mu.Lock()
	defer p.mu.Unlock()

	// Drain the available channel before closing
	for len(p.available) > 0 {
		<-p.available
	}
	close(p.available)

	// Free all contexts
	for _, ctx := range p.contexts {
		if ctx.ptr != nil {
			C.llama_wrapper_context_free(ctx.ptr)
			ctx.ptr = nil // Prevent double-free
		}
	}
	p.contexts = nil
}
