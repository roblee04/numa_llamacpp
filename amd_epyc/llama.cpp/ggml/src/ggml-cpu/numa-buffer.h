// numa-buffer.h
//
// ggml backend buffer type that allocates from the dual-NUMA allocator
// (see ggml-numa-alloc.h). When enabled, all CPU tensor allocations are
// routed through this buffer type, landing in Buffer A (Node 0). After all
// weights have been loaded, a single memcpy replicates Buffer A to Buffer B
// (Node 1). The compute hook (Phase 4) is then responsible for redirecting
// Node-1 threads to read from Buffer B.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the NUMA-replicating CPU buffer type, or nullptr if
// GGML_NUMA_REPLICATE was not compiled in or the allocator failed to init.
ggml_backend_buffer_type_t ggml_backend_cpu_numa_buffer_type(void);

// Runtime toggle: if true, the CPU device will hand out this buffer type
// as the default (instead of the plain CPU buffer type). Controlled by the
// GGML_NUMA_REPLICATE=1 environment variable or an explicit setter.
bool ggml_backend_cpu_numa_is_enabled(void);
void ggml_backend_cpu_numa_set_enabled(bool enabled);

// Finalize replication: copy Buffer A -> Buffer B. Intended to be called
// once, after the model has finished loading. Safe to call multiple times
// (idempotent beyond the first call within a load cycle).
void ggml_backend_cpu_numa_finalize_load(void);

#ifdef __cplusplus
}
#endif
