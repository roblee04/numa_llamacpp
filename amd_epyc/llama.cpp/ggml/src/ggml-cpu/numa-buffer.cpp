// numa-buffer.cpp
//
// Implementation of the NUMA-replicating CPU buffer type.
// See numa-buffer.h for design notes.

#ifdef GGML_NUMA_REPLICATE

#include "numa-buffer.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"

extern "C" {
#include "ggml-numa-alloc.h"
}

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// ---------------------------------------------------------------------------
// Runtime enable flag
// ---------------------------------------------------------------------------

static std::atomic<bool> g_numa_enabled{false};
static std::once_flag    g_numa_env_check;

static void check_env_enable() {
    const char * env = std::getenv("GGML_NUMA_REPLICATE");
    if (env && env[0] == '1') {
        g_numa_enabled.store(true);
    }
}

bool ggml_backend_cpu_numa_is_enabled(void) {
    std::call_once(g_numa_env_check, check_env_enable);
    return g_numa_enabled.load();
}

void ggml_backend_cpu_numa_set_enabled(bool enabled) {
    g_numa_enabled.store(enabled);
}

// ---------------------------------------------------------------------------
// Lazy allocator initialization
// ---------------------------------------------------------------------------

// We don't know the total tensor footprint up-front. We reserve a generously
// sized pool (default 80 GiB per node, override with GGML_NUMA_BUFFER_GB).
// mmap + mbind only reserves virtual space until pages are actually touched,
// so oversizing is cheap VM-wise, but keep in mind that numa_touch_pages() in
// the allocator will fault EVERY page — so set this only as large as you need.

static std::once_flag g_numa_alloc_init_flag;
static bool           g_numa_alloc_init_ok = false;

static void init_allocator_once() {
    size_t gb = 80;  // default
    const char * env = std::getenv("GGML_NUMA_BUFFER_GB");
    if (env) {
        char * end = nullptr;
        unsigned long v = std::strtoul(env, &end, 10);
        if (end != env && v > 0 && v < 1024) {
            gb = (size_t) v;
        }
    }
    const size_t bytes = gb * (size_t) (1ULL << 30);
    fprintf(stderr, "[numa-buffer] initializing dual-NUMA allocator: %zu GiB per node\n", gb);

    int rc = ggml_numa_alloc_init(bytes);
    if (rc != 0) {
        fprintf(stderr, "[numa-buffer] allocator init FAILED (rc=%d); "
                        "falling back to regular CPU buffer type\n", rc);
        g_numa_enabled.store(false);
        g_numa_alloc_init_ok = false;
        return;
    }
    g_numa_alloc_init_ok = true;
}

// ---------------------------------------------------------------------------
// Local copies of trivial buffer-type helpers.
//
// The functions ggml_backend_cpu_buffer_type_get_alignment() and
// ggml_backend_cpu_buffer_type_is_host() are `static` inside
// ggml-backend.cpp, so we cannot link to them from here. They are trivial
// enough to reimplement: alignment is the library-wide TENSOR_ALIGNMENT
// constant (64 bytes) and "is host" is unconditionally true for any CPU
// memory (our NUMA-pinned buffers are still host memory).
// ---------------------------------------------------------------------------

static size_t numa_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TENSOR_ALIGNMENT;
}

static bool numa_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

// ---------------------------------------------------------------------------
// Buffer-type hooks
// ---------------------------------------------------------------------------

static const char * numa_buft_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "CPU_NUMA";
}

// Buffer "free" is a no-op: we never release individual allocations from the
// bump allocator. The whole dual-pool is freed at process exit via
// ggml_numa_alloc_free() (or just left for the kernel to reclaim).
static void numa_buffer_free(ggml_backend_buffer_t buffer) {
    // No-op: bump allocator. The context pointer is into Buffer A; do NOT free.
    GGML_UNUSED(buffer);
}

static ggml_backend_buffer_t numa_buft_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {

    std::call_once(g_numa_alloc_init_flag, init_allocator_once);
    if (!g_numa_alloc_init_ok) {
        return nullptr;
    }

    const size_t align = numa_buft_get_alignment(buft);

    void * ptr = ggml_numa_alloc_node0(size, align);
    if (!ptr) {
        GGML_LOG_ERROR("[numa-buffer] failed to allocate %zu bytes from NUMA pool "
                       "(used=%zu, cap=%zu)\n",
                       size,
                       ggml_numa_alloc_bytes_used(),
                       ggml_numa_alloc_bytes_capacity());
        return nullptr;
    }

    // Build a standard CPU buffer wrapping our NUMA-pinned pointer. We reuse
    // the CPU iface for tensor initialization, copy, memset, etc. — all of
    // which are safe because the pointer is a regular CPU-addressable address.
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft              = buft;
    buffer->iface.free_buffer = numa_buffer_free;  // override: no free

    return buffer;
}

// ---------------------------------------------------------------------------
// Buffer type singleton
// ---------------------------------------------------------------------------

ggml_backend_buffer_type_t ggml_backend_cpu_numa_buffer_type(void) {
    static struct ggml_backend_buffer_type type = {
        /* .iface   = */ {
            /* .get_name         = */ numa_buft_get_name,
            /* .alloc_buffer     = */ numa_buft_alloc_buffer,
            /* .get_alignment    = */ numa_buft_get_alignment,
            /* .get_max_size     = */ nullptr,
            /* .get_alloc_size   = */ nullptr,
            /* .is_host          = */ numa_buft_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &type;
}

// ---------------------------------------------------------------------------
// Finalization: A -> B replication
// ---------------------------------------------------------------------------

void ggml_backend_cpu_numa_finalize_load(void) {
    if (!g_numa_alloc_init_ok) return;
    ggml_numa_replicate_to_node1();
}

#else  // !GGML_NUMA_REPLICATE -------------------------------------------------

#include "numa-buffer.h"

extern "C" {

ggml_backend_buffer_type_t ggml_backend_cpu_numa_buffer_type(void)    { return nullptr; }
bool ggml_backend_cpu_numa_is_enabled(void)                           { return false;  }
void ggml_backend_cpu_numa_set_enabled(bool enabled)                  { (void) enabled; }
void ggml_backend_cpu_numa_finalize_load(void)                        {}

}  // extern "C"

#endif  // GGML_NUMA_REPLICATE
