// ggml-numa-alloc.h
//
// NUMA-aware dual-buffer allocator for weight replication across sockets.
//
// Design:
//   - Two equally-sized buffers are allocated, one pinned to NUMA node 0 and
//     one pinned to NUMA node 1 via mbind(MPOL_BIND).
//   - Both buffers share an identical internal layout: an allocation at
//     offset `k` in Buffer A lives at offset `k` in Buffer B.
//   - This lets the compute hook translate a Node-0 pointer to its Node-1
//     mirror (or vice versa) with a single add/sub, avoiding any map lookup
//     on the hot path.
//   - After loading weights into Buffer A, a one-time memcpy replicates the
//     contents to Buffer B.
//
// This module is only compiled when GGML_NUMA_REPLICATE is defined.

#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the dual-buffer NUMA allocator.
//
// `size` is the number of bytes reserved on EACH node (so total RSS ~= 2*size).
// Typical value: just over the model weight size, e.g. 72 GB for a 70 GB Q8 model.
//
// Returns 0 on success, negative errno-style code on failure.
// On failure no buffers are allocated and subsequent calls are no-ops.
int  ggml_numa_alloc_init(size_t size);

// Release both buffers. Safe to call even if init failed or was never called.
void ggml_numa_alloc_free(void);

// Returns true if the allocator is initialized and both buffers are ready.
bool ggml_numa_alloc_is_ready(void);

// Bump-allocate `nbytes` from Buffer A (Node 0), aligned to `align` bytes.
// `align` must be a power of two (e.g. 64 for cache-line, 4096 for page).
//
// Returns a pointer into Buffer A on success, NULL if out of space or not
// initialized. The returned pointer can be converted to the Node-1 mirror
// address via ggml_numa_mirror_to_node1().
void * ggml_numa_alloc_node0(size_t nbytes, size_t align);

// Translate a pointer inside Buffer A to the equivalent address inside
// Buffer B (Node 1). If `ptr` is NULL or outside Buffer A, returns `ptr`
// unchanged (safe fallback).
void * ggml_numa_mirror_to_node1(const void * ptr_in_buffer_a);

// Inverse of the above: translate a Buffer-B pointer back to Buffer A.
void * ggml_numa_mirror_to_node0(const void * ptr_in_buffer_b);

// One-shot: copy the currently-used portion of Buffer A into Buffer B.
// Called exactly once after all weights have been loaded into Buffer A.
void ggml_numa_replicate_to_node1(void);

// Introspection helpers (useful for numastat verification and logging).
size_t ggml_numa_alloc_bytes_used(void);      // total bytes handed out from Buffer A
size_t ggml_numa_alloc_bytes_capacity(void);  // size of each buffer
void * ggml_numa_alloc_buffer_node0(void);    // base of Buffer A
void * ggml_numa_alloc_buffer_node1(void);    // base of Buffer B

#ifdef __cplusplus
}
#endif
