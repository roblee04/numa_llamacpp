# NUMA-Aware llama.cpp 
A detailed walk-through of what I built, why it works, and how it fits
into the existing llama.cpp code base. 

---

## 1.  High Level Summary

I wanted to make NUMA work more efficiently with llama.cpp. Instead of the stock open-source (one socket doing most of the work while the other starves on the interconnect.), I decided to reduce the amount of interconnect accesses by keeping weights local to the threads.

On a dual AMD EPYC 9654 (192 physical cores, 2 NUMA nodes, 1.5 TiB RAM)
the result is:

| Model              | Metric | Vanilla | Ours  | Speedup |
|--------------------|--------|--------:|------:|--------:|
| Llama 3 8B Q4      | tg128  |    37   |   53  | **1.45×** |
| Llama 3.3 70B Q8   | pp512  |    47   |   76  | **1.61×** |
| Llama 3.3 70B Q8   | tg128  |   3.84  |  6.71 | **1.75×** |

The gain comes from three changes, each building on the previous:

1. **Dual-buffer weight replication**: model weights are copied to
   both NUMA nodes at load time, so every read is local.
2. **Strict per-core thread pinning**: 192 worker threads, each locked
   to one physical core, so the Linux scheduler can't shuffle them
   across sockets mid-computation.
3. **Per-thread weight-pointer swap in the GEMM kernel**: each worker
   reads weights from the copy that lives on *its* NUMA node, with a
   single-cycle address translation on the hot path.

The changes are gated behind a
compile flag `GGML_NUMA_REPLICATE` and a runtime env var of the same
name, so the vanilla code path is untouched when the feature is off.

---

## 2. Background: llama.cpp and ggml in one page

**llama.cpp** is a C/C++ inference engine for LLaMA-family LLMs. It is
the de-facto standard for running these models on CPUs and consumer
GPUs. The project is split into three layers:

```
  llama.cpp        — model loading, tokenizer, KV cache, top-level API
     │
     ▼
  ggml             — a tensor library (think "NumPy in C").  Defines the
                     `ggml_tensor` struct and the compute graph.
     │
     ▼
  ggml-cpu         — the CPU "backend": a thread pool + per-op compute
                     kernels (mul_mat, rms_norm, softmax, rope, ...).
                     Implements the OP table that the graph walks.
```

Every CPU backend carries a **buffer type** — an abstract allocator with
an interface like:

```c
struct ggml_backend_buffer_type_i {
    const char *            (*get_name)(ggml_backend_buffer_type_t);
    ggml_backend_buffer_t   (*alloc_buffer)(ggml_backend_buffer_type_t, size_t);
    size_t                  (*get_alignment)(ggml_backend_buffer_type_t);
    bool                    (*is_host)(ggml_backend_buffer_type_t);
    ...
};
```

The default CPU buffer type just calls `aligned_alloc` and has no idea
which NUMA node the memory lands on. **Adding a new buffer type is the
clean, supported extension point for NUMA replication.**

The CPU backend's compute dispatcher (`ggml_compute_forward_mul_mat` and
friends) is called once per op per graph-execution. It is a plain
C function that takes a `ggml_compute_params` describing which thread
(`ith`) out of how many (`nth`) is calling it, and reads tensor data via
`src->data` — a raw `void *`. **This is the other extension point: if I
can swap that pointer per thread, I can redirect reads.**

---

## 3. Background: NUMA and why it matters

A modern dual-socket server is *not* a uniform memory system. Each
socket has its own DRAM controllers. Memory attached to socket 0 is
"local" to cores on socket 0 and "remote" for cores on socket 1 (and
vice-versa). Remote reads go across a coherent interconnect — AMD
Infinity Fabric (UPI on Intel) — which has several real costs:

| Property | Local DRAM | Remote DRAM |
|---|---|---|
| Latency (ACPI SLIT distance) | 10 | 32 |
| Bandwidth per socket | ~460 GB/s | ~80 GB/s across the link |
| Access also competes for | your own L3 | *both* sockets' link bandwidth |

For a 70 GB model in Q8 (8-bit), every token generated requires reading
the entire 70 GB at least once (that is what "memory-bound" means). On a
dual-socket machine with vanilla allocation:

- The model ends up wherever Linux's first-touch policy put it —
  typically 50/50 split by luck, often 90/10 by malloc-arena accident.
- Worker threads are free to migrate between sockets, so even if you
  started with perfect locality the scheduler will ruin it within a few
  iterations.
- Net result: ~50% of weight reads cross Infinity Fabric. That fabric
  link caps the useful memory bandwidth of the whole machine at roughly
  *one* socket's worth, even though you paid for two.

**If I put a full copy of the model on each socket and pin workers to
their socket's copy, every read is local, interconnect traffic drops to
zero, and effective bandwidth doubles.** That is the entire thesis.

The cost is 2× the RAM. On this machine I have 1.5 TiB, so a 70 GB
model → 144 GB of resident weights is trivially affordable.

---

## 4. What vanilla llama.cpp does on this hardware

Run `llama-bench` on the 70B model with no changes:

```
numastat -p $(pgrep llama-bench)
  Private   Node 0 = 25 GB    Node 1 = 45 GB    (first-touch luck)

/proc/<tid>/status  Cpus_allowed_list:  0-383    (any core, any time)
```

- Memory placement is accidental and imbalanced.
- All 192 workers are marked allowed on all 384 logical cores, so the
  Linux scheduler migrates them freely. `perf stat` shows high
  `context-switches` and `cpu-migrations` numbers.
- Every mul_mat kernel ends up reading 50-ish% of its weights over the
  interconnect.

Measured on our hardware: **3.84 tokens/second** generation on 70B Q8.

---

## 5. My changes, in one picture

```
                    ┌──── NUMA Node 0 ────┐     ┌──── NUMA Node 1 ────┐
                    │                     │     │                     │
                    │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
   weight reads →   │ │    Buffer A     │ │     │ │    Buffer B     │ │
   (local only)     │ │  72 GiB mbind   │ │     │ │  72 GiB mbind   │ │
                    │ │  Node 0 strict  │ │     │ │  Node 1 strict  │ │
                    │ └───────▲─────────┘ │     │ └───────▲─────────┘ │
                    │         │           │     │         │           │
                    │   Workers 0..95     │     │   Workers 96..191   │
                    │   pinned to cores   │     │   pinned to cores   │
                    │   0..95             │     │   96..191           │
                    │   (strict, 1 core   │     │   (strict, 1 core   │
                    │    per worker)      │     │    per worker)      │
                    └─────────────────────┘     └─────────────────────┘
                            ▲                             ▲
                            │  one-time memcpy after load │
                            └─────────────────────────────┘
                                    (replication)
```

On the hot path (inside a GEMM), each worker just does:

```
  node   = threadpool->workers[ith].numa_node_id;   // set once at startup
  src0   = (node == 1) ? mirror_to_node1(tensor->data)  // add/sub only
                       :  tensor->data;
  ... SIMD over src0 as usual ...
```

No locks, no branches per cache line, no lookup tables. The mirror
translation is `base_B + (ptr - base_A)` — a single add.

---

## 6. Implementation



### Build system

Added `option(GGML_NUMA_REPLICATE ... OFF)` to `ggml/CMakeLists.txt`,
linked libnuma, wired the new sources into the CPU backend CMake list.


Also forced `-DGGML_CPU_REPACK=OFF` because the stock repack.cpp path
wraps weights in a second, in-place-transposed buffer that bypasses our
allocator. Making repack.cpp NUMA-aware is deferred.

### Dual-buffer allocator (`ggml-numa-alloc.[ch]`)

Two nearly-identical memory regions, one per NUMA node. The allocator's
public API is small:

```c
int  ggml_numa_alloc_init(size_t size);                  // init 2× buffers
void * ggml_numa_alloc_node0(size_t nbytes, size_t a);   // bump-allocate from Buffer A
void * ggml_numa_mirror_to_node1(const void * p);        // translate A -> B
void   ggml_numa_replicate_to_node1(void);               // one-time memcpy
```

Implementation highlights:

1. **`mmap` two anonymous regions** of `size` bytes each. At this point
   no physical pages are mapped yet.
2. **`mbind(ptr, size, MPOL_BIND, nodemask, ...)`** on each region,
   with `nodemask` set to exactly one node. `MPOL_BIND` is *strict*:
   allocations that can't be satisfied on that node fail outright rather
   than falling back to another node. This is what I want.
3. **Force page fault-in immediately.** Write a single byte to every
   4 KiB page. Without this, `mbind` is lazy so pages are allocated on
   first touch, and if the first-touching thread happens to be on the
   wrong node, kernel policy is free to ignore the binding in some
   configurations. By faulting during init I make physical placement
   deterministic.
4. **Bump allocator (no free).** GGUF weight loading is append-only; I
   never free a weight tensor during inference. A monotonic atomic
   `g_used_bytes` is all I need. Alignment is passed in per-allocation
   and defaults to 64 bytes (cache line).
5. **Identical-offset layout.** Crucially, the same byte offset `k`
   refers to the same tensor in both buffers. That means mirror
   translation is a single subtract-and-add:

   ```c
   void * ggml_numa_mirror_to_node1(const void * p) {
       return (char*)g_buffer_node1 + ((char*)p - (char*)g_buffer_node0);
   }
   ```

   If `p` is outside Buffer A, the function returns it unchanged so
   non-replicated tensors (activations, KV cache, biases that happen to
   land in the regular CPU buffer type) pass through safely.

**`numa-buffer.cpp`** wraps the bump allocator in a `ggml_backend_buffer_type`
so it plugs into ggml's extension point cleanly. The buffer type:

- lazily initialises on its first `alloc_buffer` call, by this point I
  know how big the model is (from GGUF meta), so I can size the buffer
  from `GGML_NUMA_BUFFER_GB` env var
- reports `is_host = true` so llama.cpp treats these tensors as
  CPU-readable (same as the regular CPU buffer)
- on `free_buffer`, is a no-op (bump allocator doesn't free individual
  allocations)

The hook from the CPU device into the buffer type:

```cpp
// ggml/src/ggml-cpu/ggml-cpu.cpp
static ggml_backend_buffer_type_t device_get_buffer_type(...) {
#ifdef GGML_NUMA_REPLICATE
    if (ggml_backend_cpu_numa_is_enabled()) {
        return ggml_backend_cpu_numa_buffer_type();
    }
#endif
    return ggml_backend_cpu_buffer_type();
}
```

And the finalize call (copy A → B) is threaded through llama.cpp via a
**proc-address table pattern**, llama-model.cpp gets a function pointer
from the backend registry rather than `#include`-ing our header
directly. This keeps cross-module dependencies out of llama.cpp proper:

```cpp
// src/llama-model.cpp, inside load_tensors()
auto finalize = (void(*)())
    ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_numa_replicate_finalize");
if (finalize) finalize();
```

**Validation:** `numastat -p` during inference shows `Private` memory
split ~8200/8200 MiB across the two nodes. `Numa_Foreign` stays at 0 —
the signature of a clean mbind.

### Thread pinning (`ggml-cpu.c`)

llama.cpp already has thread-affinity functionality ( `set_numa_thread_affinity`,
a `ggml_threadpool` struct with per-worker cpumasks, a `strict_cpu`
flag ) but it was only used when the user explicitly passed
`--numa isolate --cpu-mask 0xFFF... --cpu-strict 1` on the command line.
Most users never set those, so the pool runs unpinned.

I added three things:

1. **A new NUMA strategy enum value, `GGML_NUMA_STRATEGY_REPLICATE = 5`**,
   in `ggml-cpu.h`. `set_numa_thread_affinity` has a switch on strategy
; adding a new case that *early-returns* preserves the strict
   per-worker pin that I set up elsewhere. Otherwise the existing cases
   would go over the pin with a looser node-level mask.

2. **A single auto-config chokepoint inside `ggml_threadpool_new_impl`.**
   This function is called by *everything* that creates a threadpool,
   both the long-lived one created by `common.cpp` and the short-lived
   disposable one created inside `ggml_graph_plan`. Pushing the
   auto-config one level deeper in the call stack catches both:

   ```c
   // pseudocode of what the chokepoint does
   if (getenv("GGML_NUMA_REPLICATE") == "1") {
       tpp->n_threads    = 192;
       tpp->strict_cpu   = true;
       tpp->numa_strategy = GGML_NUMA_STRATEGY_REPLICATE;
       for (i = 0; i < 192; i++) {
           int core = physical_core_id(i);   // skip SMT siblings
           CPU_ZERO(&tpp->cpumask[i]);
           CPU_SET (core, &tpp->cpumask[i]);
       }
       fprintf(stderr, "[numa-replicate] threadpool: 192-thread strict pin across physical cores 0..191\n");
   }
   ```

   Previously, running with `common.cpp`'s path only would miss
   `llama-cli` (which builds its own threadpool through `ggml_graph_plan`
   for single-shot inference). Doing it here catches both.

3. **Per-worker NUMA-node tagging.** Each `worker` struct gets a
   `numa_node_id` computed once at pool creation (from its CPU mask),
   and a new accessor:

   ```c
   int ggml_threadpool_worker_numa_node(struct ggml_threadpool * tp, int ith) {
       return tp->workers[ith].numa_node_id;
   }
   ```


**Validation:** `/proc/<tid>/status` on each worker shows a single-bit
`Cpus_allowed_list`:

```bash
# 192 distinct per-worker masks, one each for cores 0..191
for tid in $(ls /proc/$(pgrep -x llama-bench)/task); do
    awk '/Cpus_allowed_list/ {print}' /proc/$pid/task/$tid/status
done | sort -u | wc -l    # expect 192
```

###  Compute hook (`ggml-cpu.c`)

The actual inner-loop change is small but important. Inside
`ggml_compute_forward_mul_mat_one_chunk`:

```c
// BEFORE
const char * src0_row = (const char *) src0->data
                         + (0 + i02*nb02 + i03*nb03);

// AFTER
const char * src0_data = (const char *) src0->data;
#ifdef GGML_NUMA_REPLICATE
if (params->threadpool && ggml_numa_alloc_is_ready()) {
    if (ggml_threadpool_worker_numa_node(params->threadpool, params->ith) == 1) {
        src0_data = (const char *) ggml_numa_mirror_to_node1(src0->data);
    }
}
#endif
const char * src0_row = src0_data + (0 + i02*nb02 + i03*nb03);
```

Reading that code:

- If the feature is off at compile time, the #ifdef block disappears
  entirely → zero overhead in the vanilla build.
- If the allocator never initialised (`is_ready` is false, e.g. user
  ran with `GGML_NUMA_REPLICATE=0`), the branch collapses and I fall
  through to the original pointer means zero overhead in the vanilla
  *runtime*.
- Only when (a) the feature is compiled in, (b) the allocator is live,
  and (c) this particular worker is on Node 1 do I rewrite the
  pointer. The rewrite is a pointer subtract + add, nothing more.
- For tensors whose `src0->data` is *not* inside Buffer A
  (intermediate activations, KV cache, repacked tensors),
  `ggml_numa_mirror_to_node1` detects the pointer is out of range and
  returns it unchanged. So the hook is safe even when a single graph
  mixes replicated and non-replicated tensors.

The same change is applied in the sibling `ggml_compute_forward_mul_mat`
for the `GGML_USE_LLAMAFILE` path (which calls into `llamafile_sgemm`).
Two call sites, one local variable, identical pattern.

**Why only `mul_mat`?** Because llama.cpp's weight-bearing ops are
dominated by matrix multiplies. Attention and FFN both end up in
mul_mat. RMS-norm, softmax, rope etc. are compute-light and
activation-sized, so remote reads there are a rounding error. The
compute hook is a targeted ~10-line change, not a sprawling rewrite of
the kernel library.

**Validation:** `numastat` during a tg run shows both Node 0 and
Node 1 `Private` memory saturated (8200/8200 MiB for the 8B test
model), not just Node 0. Tokens per second scale as described above.

---

## Runtime flow, end-to-end

What actually happens when you run

```bash
GGML_NUMA_REPLICATE=1 GGML_NUMA_BUFFER_GB=72 \
  ./llama-cli -m llama-3.3-70B-q8.gguf --no-mmap -p "Hello"
```

```
  ┌──────────────────────────────────────────────────────────────┐
  │ T+0.0s   llama_init_from_model                               │
  │          └─ llama_numa_init sees env var, promotes           │
  │             numa_strategy DISABLED→REPLICATE, calls          │
  │             numa_available() so g_state.numa.nodes[] is      │
  │             populated before any threadpool is built.        │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │ T+0.1s   GGUF open, metadata parsed, tensor list built       │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │ T+0.2s   First call to cpu_device_get_buffer_type            │
  │          returns ggml_backend_cpu_numa_buffer_type()         │
  │          → buffer type's alloc_buffer is invoked             │
  │          → lazy init fires: ggml_numa_alloc_init(72 GiB)     │
  │               - mmap 72 GiB anon  (Node 0 target)            │
  │               - mbind(MPOL_BIND, node=0)                     │
  │               - touch every page -> 72 GiB physically on N0  │
  │               - mmap 72 GiB anon  (Node 1 target)            │
  │               - mbind(MPOL_BIND, node=1)                     │
  │               - touch every page -> 72 GiB physically on N1  │
  │          [logs]                                              │
  │            [numa-alloc] faulting in 73728 MiB on node 0...   │
  │            [numa-alloc] faulting in 73728 MiB on node 1...   │
  │            [numa-alloc] ready: 73728 MiB x 2 buffers         │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │ T+90s    GGUF weight read loop                               │
  │          For each tensor:                                    │
  │            - ggml_numa_alloc_node0(size, align)              │
  │            - fread(ptr, ...) from disk into Buffer A         │
  │          At the end, Buffer A holds all 70 GB of weights.    │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │ T+~180s  llama_model_numa_replicate_finalize_if_enabled()    │
  │          (called from load_tensors after the mmap block)     │
  │          → ggml_numa_replicate_to_node1()                    │
  │          → memcpy(Buffer B, Buffer A, bytes_used)            │
  │          [logs]                                              │
  │            [numa-alloc] replicating 71494 MiB from node 0    │
  │                        to node 1...                          │
  │            [numa-alloc] replication complete                 │
  │          Both nodes now hold an identical, byte-for-byte     │
  │          copy of the weights at identical offsets.           │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │ T+~180s  First ggml_graph_compute                            │
  │          → ggml_graph_plan builds a disposable threadpool    │
  │          → inside ggml_threadpool_new_impl, the env-var      │
  │            chokepoint sees GGML_NUMA_REPLICATE=1, so:        │
  │              - n_threads := 192                              │
  │              - strict_cpu := true                            │
  │              - cpumask[i] := {core i} for i in 0..191        │
  │              - each worker's numa_node_id computed from      │
  │                its cpumask                                   │
  │              - deque init per worker (Phase-4 scaffold)      │
  │          [logs]                                              │
  │            [numa-replicate] threadpool: 192-thread strict    │
  │                        pin across physical cores 0..191      │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │ T+~180s  For each op in the graph:                           │
  │            each worker calls its op's compute fn with        │
  │              params->ith    = its worker index               │
  │              params->nth    = 192                            │
  │              params->threadpool = the pool (Phase-3 exposed) │
  │          Inside ggml_compute_forward_mul_mat_one_chunk:      │
  │            numa_node = threadpool->workers[ith].numa_node_id │
  │            src0_data = (numa_node == 1) ?                    │
  │                mirror_to_node1(src0->data) :                 │
  │                src0->data                                    │
  │            → 100% of weight reads are local DRAM             │
  │            → ~zero bytes cross Infinity Fabric               │
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │ T+X      Token emitted. Loop for next token.                 │
  └──────────────────────────────────────────────────────────────┘
```

---

## 8. File-by-file index

New files (all under `ggml/src/ggml-cpu/`):

| File | Role |
|------|------|
| `ggml-numa-alloc.h` | Dual-buffer allocator public API (C, link-safe) |
| `ggml-numa-alloc.c` | `mmap` + `mbind(MPOL_BIND)` + per-page touch + bump allocator + mirror translation |
| `numa-buffer.h` | ggml backend buffer type declaration |
| `numa-buffer.cpp` | Buffer-type implementation; lazy init; honours `GGML_NUMA_REPLICATE` and `GGML_NUMA_BUFFER_GB` |

Modified files:

| File | Change |
|------|--------|
| `ggml/CMakeLists.txt` | `option(GGML_NUMA_REPLICATE)` flag |
| `ggml/src/ggml-cpu/CMakeLists.txt` | `find_library(numa)`, link libnuma, new sources |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | Buffer type dispatch; proc-address entries |
| `ggml/src/ggml-cpu/ggml-cpu.c` | Core C runtime: new strategy enum case; threadpool auto-config chokepoint; per-worker NUMA-node tag; `ggml_threadpool_worker_numa_node` getter; compute hook in mul_mat (Phase 5) |
| `ggml/include/ggml-cpu.h` | Public API additions |
| `src/llama.cpp` | Promote `numa_strategy` to REPLICATE when env var is set |
| `src/llama-model.cpp` | Call `ggml_backend_cpu_numa_replicate_finalize` after load |
| `common/common.cpp` | Comment-only (auto-config lives in the chokepoint below it) |

---

## 9. Known limitations and next steps

Documented here because they shape how the next user of this code
should think about "what have I got, what am I still paying for?":

1. **`--no-mmap` required.** The mmap code path uses a different
   allocation strategy (`buffer_from_host_ptr`, type `CPU_Mapped`) that
   sidesteps the CPU device buffer-type hook. Our allocator is never
   called and weights end up OS-scattered. Fix: intercept or replicate
   `buffer_from_host_ptr`. Needed for any production deployment where
   paging behaviour matters.
2. **`GGML_CPU_REPACK=OFF` required.** The stock repack module takes
   the as-loaded weight tensor and builds a transposed copy optimised
   for AVX-512 GEMV. It allocates via a separate path that bypasses us.
   Fix: teach `repack.cpp` to allocate through `ggml_numa_alloc_node0`
   and participate in replication.
3. **KV cache and activations still on Node 0.** My implementation replicates
   *weights* only. For long contexts (tg > 512 on 70B), the KV cache
   grows to gigabytes and Node-1 workers pay remote-read cost on every
   attention step. This is the dominant ceiling on long-context
   generation — see the tg512 vs tg128 speedup drop on 70B
   (1.18× vs 1.75×). Replicating or sharding the KV cache across nodes
   is the obvious next optimisation.
4. **Dead-code warnings** (`ggml_numa_alloc_on_node` nested-function,
   `ggml_pin_thread_to_core` missing prototype) — pre-existing, not
   blocking, should be cleaned up.

---

## 10. How to reproduce / use the engine

Build:

```bash

cd amd_epyc/llama.cpp
cmake -B build \
  -DGGML_NUMA_REPLICATE=ON \
  -DGGML_CPU_REPACK=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 
```

Run (the env var gates the entire feature):

```bash
export GGML_NUMA_REPLICATE=1    # turn on allocator + pin + compute hook
export GGML_NUMA_BUFFER_GB=72   # per-node buffer (≥ model size)

./build/bin/llama-cli -m llama-3.3-70B.gguf \
  --no-mmap --single-turn -p "..." -n 128
```

Expected startup banner:

```
[numa-buffer] initializing dual-NUMA allocator: 72 GiB per node
[numa-alloc] faulting in 73728 MiB on node 0...
[numa-alloc] faulting in 73728 MiB on node 1...
[numa-alloc] ready: 73728 MiB x 2 buffers (node0=..., node1=...)
[numa-alloc] replicating 71494 MiB from node 0 to node 1...
[numa-alloc] replication complete
[numa-replicate] threadpool: 192-thread strict pin across physical cores 0..191
```

Setting `GGML_NUMA_REPLICATE=0` on the same binary disables the
feature cleanly, no buffer is allocated, no pin is applied, the
compute hook is a no-op, and the code paths are effectively vanilla.
This was used as the A/B baseline in the companion metrics.

---

## 11. Design choices worth highlighting

A few choices that are non-obvious but intentional:

- **Env-var gating over compile-time only.** I want one binary to
  cover both the "on" and "off" case so the A/B comparison is
  apples-to-apples (same compiler flags, same linkage, same optimiser
  output). A compile-time gate only would have required two binaries
  and introduced confounds.
- **Buffer-type extension point, not a malloc override.** llama.cpp's
  backend abstraction exists precisely for this kind of customisation.
  Hooking at the abstraction layer means I don't have to patch the
  allocator everywhere ggml uses memory, and I inherit the correct
  alignment / accessor semantics for free.
- **Mirror translation by pointer arithmetic, not table lookup.**
  Identical-offset layout turns the redirect into a single
  add/subtract. A hash table of pointer → mirror would work too, but
  would cost a cache line per tensor and a load per mul_mat, neither
  of which I can afford on the hot path.
- **Chokepoint in `ggml_threadpool_new_impl`, not in `common.cpp`.**
  Two separate call sites create threadpools. Putting the
  auto-configuration one level deeper catches both and eliminates a
  whole class of "it works in one binary but not the other" bugs.
- **Proc-address table for the finalize callback.** Keeps
  `llama-model.cpp` free of NUMA-specific headers. Follows the same
  pattern llama.cpp already uses for optional backend features.
- **Safe fallback in `mirror_to_node1`.** Returning `ptr` unchanged for
  any out-of-range pointer means I never corrupt state when the hook
  fires on a tensor I don't own. That's what lets us drop the hook
  into a generic kernel without surrounding it in taxonomical checks.

---
