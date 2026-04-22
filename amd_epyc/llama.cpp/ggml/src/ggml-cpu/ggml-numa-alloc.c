// ggml-numa-alloc.c
//
// Implementation of the dual-buffer NUMA allocator.
// See ggml-numa-alloc.h for design notes.

#include "ggml-numa-alloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdatomic.h>
#include <stdint.h>

#if defined(GGML_NUMA_REPLICATE)

#include <sys/mman.h>
#include <unistd.h>

// libnuma's public header triggers -Wcast-qual warnings; they're harmless
// and we can't fix them upstream, so locally suppress.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <numa.h>
#include <numaif.h>
#pragma GCC diagnostic pop

// ---------------------------------------------------------------------------
// Internal state (single global allocator instance)
// ---------------------------------------------------------------------------

static void *            g_buffer_node0 = NULL;  // base of Buffer A
static void *            g_buffer_node1 = NULL;  // base of Buffer B
static size_t            g_buffer_size  = 0;     // bytes per buffer
static atomic_size_t     g_used_bytes   = 0;     // bump offset (bytes consumed in Buffer A)
static bool              g_initialized  = false;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int numa_bind_buffer(void * ptr, size_t size, int node) {
    // Build a nodemask with just the target node bit set.
    // We size the mask generously (up to 64 nodes) which is more than enough
    // for any dual-socket system.
    unsigned long nodemask[16] = {0};  // 16 * 64 = 1024 nodes max
    const int max_node = (int)(sizeof(nodemask) * 8);

    if (node < 0 || node >= max_node) {
        fprintf(stderr, "[numa-alloc] invalid node id %d\n", node);
        return -EINVAL;
    }

    nodemask[node / 64] |= (1UL << (node % 64));

    // MPOL_BIND = strict: allocations MUST come from the specified node, else fail.
    // maxnode is the number of bits in the mask (not bytes). Pass one past the
    // highest valid node bit we might touch.
    if (mbind(ptr, size, MPOL_BIND, nodemask, max_node, 0) != 0) {
        fprintf(stderr, "[numa-alloc] mbind(node=%d) failed: %s\n", node, strerror(errno));
        return -errno;
    }

    return 0;
}

static void numa_touch_pages(void * ptr, size_t size) {
    // Force the kernel to physically allocate every page on the bound node.
    // Without this, pages remain unmapped until first access, and may end up
    // on the wrong node if the first-touching thread is on a different node.
    //
    // Touching every page at 4 KiB stride is the standard idiom. memset()
    // also works, but explicit per-page writes make intent clearer and avoid
    // any chance of the compiler eliding the store.
    const size_t page_size = 4096;
    volatile char * p = (volatile char *) ptr;
    for (size_t off = 0; off < size; off += page_size) {
        p[off] = 0;
    }
}

static inline size_t align_up(size_t value, size_t align) {
    return (value + align - 1) & ~(align - 1);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

int ggml_numa_alloc_init(size_t size) {
    if (g_initialized) {
        fprintf(stderr, "[numa-alloc] already initialized\n");
        return -EALREADY;
    }

    if (numa_available() < 0) {
        fprintf(stderr, "[numa-alloc] libnuma reports NUMA is not available\n");
        return -ENOTSUP;
    }

    const int max_node = numa_max_node();
    if (max_node < 1) {
        fprintf(stderr, "[numa-alloc] need at least 2 NUMA nodes, found %d\n", max_node + 1);
        return -ENOTSUP;
    }

    // Round up to page size so mbind has clean boundaries.
    const size_t page_size = (size_t) sysconf(_SC_PAGESIZE);
    size = align_up(size, page_size);

    // Reserve virtual address space for both buffers. MAP_ANONYMOUS | MAP_PRIVATE
    // gives us zero-filled anonymous memory; pages are not yet backed by RAM.
    const int prot  = PROT_READ | PROT_WRITE;
    const int flags = MAP_PRIVATE | MAP_ANONYMOUS;

    void * buf0 = mmap(NULL, size, prot, flags, -1, 0);
    if (buf0 == MAP_FAILED) {
        fprintf(stderr, "[numa-alloc] mmap(node0, %zu bytes) failed: %s\n",
                size, strerror(errno));
        return -errno;
    }

    void * buf1 = mmap(NULL, size, prot, flags, -1, 0);
    if (buf1 == MAP_FAILED) {
        fprintf(stderr, "[numa-alloc] mmap(node1, %zu bytes) failed: %s\n",
                size, strerror(errno));
        munmap(buf0, size);
        return -errno;
    }

    // Bind each buffer strictly to its NUMA node.
    int rc = numa_bind_buffer(buf0, size, 0);
    if (rc != 0) {
        munmap(buf0, size);
        munmap(buf1, size);
        return rc;
    }

    rc = numa_bind_buffer(buf1, size, 1);
    if (rc != 0) {
        munmap(buf0, size);
        munmap(buf1, size);
        return rc;
    }

    // Force physical page allocation NOW (on the correct nodes).
    // This is the single most important step: mbind() alone is lazy.
    fprintf(stderr, "[numa-alloc] faulting in %zu MiB on node 0...\n", size >> 20);
    numa_touch_pages(buf0, size);
    fprintf(stderr, "[numa-alloc] faulting in %zu MiB on node 1...\n", size >> 20);
    numa_touch_pages(buf1, size);

    g_buffer_node0 = buf0;
    g_buffer_node1 = buf1;
    g_buffer_size  = size;
    atomic_store(&g_used_bytes, 0);
    g_initialized  = true;

    fprintf(stderr, "[numa-alloc] ready: %zu MiB x 2 buffers (node0=%p, node1=%p)\n",
            size >> 20, buf0, buf1);
    return 0;
}

void ggml_numa_alloc_free(void) {
    if (!g_initialized) {
        return;
    }
    if (g_buffer_node0) munmap(g_buffer_node0, g_buffer_size);
    if (g_buffer_node1) munmap(g_buffer_node1, g_buffer_size);
    g_buffer_node0 = NULL;
    g_buffer_node1 = NULL;
    g_buffer_size  = 0;
    atomic_store(&g_used_bytes, 0);
    g_initialized  = false;
}

bool ggml_numa_alloc_is_ready(void) {
    return g_initialized;
}

void * ggml_numa_alloc_node0(size_t nbytes, size_t align) {
    if (!g_initialized || nbytes == 0) {
        return NULL;
    }
    if (align == 0) align = 64;  // default to cache-line alignment

    // Atomic bump allocator: reserve [old_used, new_used) in one CAS loop.
    // This is safe to call from multiple threads during model load, though in
    // practice GGUF loading is single-threaded.
    size_t old_used, new_used, aligned_off;
    do {
        old_used    = atomic_load(&g_used_bytes);
        aligned_off = align_up(old_used, align);
        new_used    = aligned_off + nbytes;
        if (new_used > g_buffer_size) {
            fprintf(stderr, "[numa-alloc] OOM: requested %zu (aligned %zu), "
                            "used %zu, capacity %zu\n",
                    nbytes, aligned_off, old_used, g_buffer_size);
            return NULL;
        }
    } while (!atomic_compare_exchange_weak(&g_used_bytes, &old_used, new_used));

    return (char *) g_buffer_node0 + aligned_off;
}

void * ggml_numa_mirror_to_node1(const void * ptr) {
    // The `const` on the parameter documents that we don't write through
    // `ptr`; the returned pointer is writable by design (it's the caller's
    // same buffer, just on the other node). Cast via uintptr_t to drop const
    // cleanly without triggering -Wcast-qual.
    if (!g_initialized || ptr == NULL) {
        return (void *) (uintptr_t) ptr;
    }
    const char * base = (const char *) g_buffer_node0;
    const char * p    = (const char *) ptr;
    if (p < base || p >= base + g_buffer_size) {
        // Pointer wasn't from our allocator; return unchanged.
        return (void *) (uintptr_t) ptr;
    }
    return (char *) g_buffer_node1 + (p - base);
}

void * ggml_numa_mirror_to_node0(const void * ptr) {
    if (!g_initialized || ptr == NULL) {
        return (void *) (uintptr_t) ptr;
    }
    const char * base = (const char *) g_buffer_node1;
    const char * p    = (const char *) ptr;
    if (p < base || p >= base + g_buffer_size) {
        return (void *) (uintptr_t) ptr;
    }
    return (char *) g_buffer_node0 + (p - base);
}

void ggml_numa_replicate_to_node1(void) {
    if (!g_initialized) {
        return;
    }
    const size_t used = atomic_load(&g_used_bytes);
    fprintf(stderr, "[numa-alloc] replicating %zu MiB from node 0 to node 1...\n",
            used >> 20);
    memcpy(g_buffer_node1, g_buffer_node0, used);
    fprintf(stderr, "[numa-alloc] replication complete\n");
}

size_t ggml_numa_alloc_bytes_used(void) {
    return atomic_load(&g_used_bytes);
}

size_t ggml_numa_alloc_bytes_capacity(void) {
    return g_buffer_size;
}

void * ggml_numa_alloc_buffer_node0(void) { return g_buffer_node0; }
void * ggml_numa_alloc_buffer_node1(void) { return g_buffer_node1; }

#else  // !GGML_NUMA_REPLICATE --------------------------------------------------

// Stubs so callers can link unconditionally when the feature is disabled.

int    ggml_numa_alloc_init(size_t size)                  { (void) size; return -ENOTSUP; }
void   ggml_numa_alloc_free(void)                         {}
bool   ggml_numa_alloc_is_ready(void)                     { return false; }
void * ggml_numa_alloc_node0(size_t n, size_t a)          { (void) n; (void) a; return NULL; }
void * ggml_numa_mirror_to_node1(const void * p)          { return (void *) (uintptr_t) p; }
void * ggml_numa_mirror_to_node0(const void * p)          { return (void *) (uintptr_t) p; }
void   ggml_numa_replicate_to_node1(void)                 {}
size_t ggml_numa_alloc_bytes_used(void)                   { return 0; }
size_t ggml_numa_alloc_bytes_capacity(void)               { return 0; }
void * ggml_numa_alloc_buffer_node0(void)                 { return NULL; }
void * ggml_numa_alloc_buffer_node1(void)                 { return NULL; }

#endif  // GGML_NUMA_REPLICATE
