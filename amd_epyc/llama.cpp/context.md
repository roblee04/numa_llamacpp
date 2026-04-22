# Context Handoff — NUMA-Aware llama.cpp Engine

## Project Overview

Implementing a NUMA-aware memory & threading engine for llama.cpp on **dual AMD EPYC 9654 (192 physical cores, 2 NUMA nodes, 500 GB RAM)** per `amd_epyc/implement.md`. Target: Llama-3.3-70B-Instruct-Q8 (~70 GB).

**6 phases total:**
1. ✅ Build system + libnuma integration
2. ✅ **Memory replication** (DONE — validated)
3. ✅ **Thread-to-core pinning** (DONE — validated, 1.9× gen speedup)
4. ⬜ Lock-free work-stealing queues
5. ✅ **GGML compute hook** (DONE — another 1.5–1.9× on top of Phase 3)
6. ⬜ Validation with `numastat` / `perf` ← NEXT (deeper perf counters)

## Environment

- **Working dir:** `/WAVE/users2/unix/rblee/numa_llamacpp/amd_epyc/llama.cpp`
- **User:** `rblee@login02.wave.scu.edu` (SCU HPC; SLURM, no sudo)
- **NUMA module:** `module load numactl/2.0.19-GCCcore-14.3.0` (sets `$EBROOTNUMACTL`)
- **Build:** `cmake --build build -j 16`
- **Test model:** `../../Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` (4685 MiB)
- **HPC running Node:** srun --partition=ycho2_lab --nodelist=ycho201 --nodes=1 --exclusive --ntasks=1 --cpus-per-task=384 --mem=500G --time=0-04:00:00 --pty /bin/bash

### modules to be loaded
module load GCC/14.3.0
module load CMake/4.0.3-GCCcore-14.3.0
module load hwloc/2.12.1-GCCcore-14.3.0
module load numactl/2.0.19-GCCcore-14.3.0


## Runtime contract (Phase 2 complete)

```bash
export GGML_NUMA_REPLICATE=1        # enable dual-buffer NUMA allocator
export GGML_NUMA_BUFFER_GB=8        # size per node (default 80)
./build/bin/llama-cli -m model.gguf --no-mmap ...    # --no-mmap required!
```

**Important:** `--no-mmap` is required. With mmap on, weights go through `buffer_from_host_ptr` (CPU_Mapped type), bypassing our NUMA allocator entirely. Eventually this needs fixing.

Also: built with `-DGGML_CPU_REPACK=OFF` to force 100% of weights through our buffer type. Re-enabling repack in production will require making `repack.cpp` NUMA-aware too.

## Phase 2 Artifacts (DO NOT LOSE)

### New files
| File | Purpose |
|------|---------|
| `ggml/src/ggml-cpu/ggml-numa-alloc.h` | Dual-buffer allocator API |
| `ggml/src/ggml-cpu/ggml-numa-alloc.c` | `mbind` + `MPOL_BIND` + forced page-fault |
| `ggml/src/ggml-cpu/numa-buffer.h` | `CPU_NUMA` buffer type decl |
| `ggml/src/ggml-cpu/numa-buffer.cpp` | Buffer type impl, lazy allocator init |

### Modified files
| File | Change |
|------|--------|
| `ggml/CMakeLists.txt` | `option(GGML_NUMA_REPLICATE ... OFF)` near GGML_OPENMP |
| `ggml/src/ggml-cpu/CMakeLists.txt` | `find_library(numa)`, linked after OPENMP block; new sources added to GGML_CPU_SOURCES |
| `ggml/src/ggml-cpu/ggml-cpu.cpp` | `#include "numa-buffer.h"` under `#ifdef GGML_NUMA_REPLICATE`; `device_get_buffer_type()` returns NUMA buft if enabled; added two `get_proc_address` entries for `ggml_backend_cpu_numa_replicate_is_enabled` and `ggml_backend_cpu_numa_replicate_finalize` |
| `src/llama-model.cpp` | Forward-decl `llama_model_numa_replicate_finalize_if_enabled()` near top (after `#include <vector>`); full definition before `arch_name()`; one-line call before `return true` in `load_tensors()` just after `if (use_mmap_buffer)` block |

### Key design decisions made
- **Lazy allocator init** on first `alloc_buffer` (sized from `GGML_NUMA_BUFFER_GB`)
- **Bump allocator** (no free) — fine for model weights
- **Identical offset layout** in Buffer A & B → mirror translation is `base_B + (ptr - base_A)`
- **Trivial helper fns** (`get_alignment`, `is_host`) reimplemented locally because the ones in `ggml-backend.cpp` are `static`
- **Proc-address table pattern** for exposing `finalize_load()` to `llama-model.cpp` without cross-module header includes

## Validation Results (Phase 2)

```
[numa-buffer] initializing dual-NUMA allocator: 8 GiB per node
[numa-alloc] faulting in 8192 MiB on node 0...
[numa-alloc] faulting in 8192 MiB on node 1...
[numa-alloc] ready: 8192 MiB x 2 buffers (node0=0x7f648c000000, node1=0x7f628c000000)
[numa-alloc] replicating 4685 MiB from node 0 to node 1...
[numa-alloc] replication complete
```

**numastat:** Node 0: 8211 MB, Node 1: 8207 MB (near-perfectly balanced)
**Memory breakdown:** 5983 MiB Host (our NUMA buft) = 4685 model + 1024 context + 274 compute. No CPU_REPACK split.
**Correctness:** Coherent output preserved. Capital-of-France test passed.

## Phase 3 Artifacts (completed this session)

### Modified files
| File | Change |
|------|--------|
| `ggml/include/ggml-cpu.h` | Added `GGML_NUMA_STRATEGY_REPLICATE = 5` enum value; added `ggml_threadpool_worker_numa_node()` API for Phase 5 to read back each worker's NUMA node |
| `ggml/src/ggml-cpu/ggml-cpu.c` | Added `ggml_numa_node_of_cpu` / `ggml_numa_node_from_cpumask` helpers (reuse `g_state.numa.nodes[]` already populated by `ggml_numa_init`); REPLICATE case in `set_numa_thread_affinity` (early return to preserve per-worker pin); env-var auto-config block in `ggml_threadpool_new_impl` — single chokepoint covering both the common.cpp path and the disposable-threadpool path in `ggml_graph_plan`; per-worker `numa_node_id` + `ggml_deque_init` wiring in both OpenMP and non-OpenMP branches; `ggml_threadpool_worker_numa_node` getter |
| `src/llama.cpp` | `llama_numa_init` promotes `DISABLED` → `REPLICATE` when `GGML_NUMA_REPLICATE=1`, so node enumeration runs before threadpool creation |

### Validation Results (Phase 3)

```
[numa-alloc] ready: 8192 MiB x 2 buffers (node0=..., node1=...)
[numa-alloc] replication complete
[numa-replicate] threadpool: 192-thread strict pin across physical cores 0..191
```

- `llama-cli`: Generation **54.0 t/s vs 28.2 t/s baseline (1.9×)**; Prompt 223 t/s vs 160 t/s (1.4×). Paris-capital test correct.
- `llama-bench` tg2048: 31.85 t/s.
- All 192 workers show single-CPU-bit masks 0..191 (verified via `/proc/<tid>/status` `Cpus_allowed`).
- `numastat`: Node 0 ≈ 8197 MB, Node 1 ≈ 8261 MB (balanced; Node 1 buffer still allocated but not dispatched to — that's Phase 5).

### Non-blocking cleanup deferred
- `ggml_numa_alloc_on_node` is defined inside `ggml_numa_init`'s body (GCC nested-function extension, ISO C forbids) — appears to be dead code from prior work. Leave warning for now.
- `ggml_pin_thread_to_core` missing prototype — prior code, same treatment.

## Phase 5 Artifacts (completed this session)

### Modified files
| File | Change |
|------|--------|
| `ggml/src/ggml-cpu/ggml-cpu.c` | `#include "ggml-numa-alloc.h"` under `#ifdef GGML_NUMA_REPLICATE`; in `ggml_compute_forward_mul_mat_one_chunk` compute a `src0_data` local that becomes the Buffer-B mirror when `ggml_threadpool_worker_numa_node(params->threadpool, ith) == 1`, and use it to derive `src0_row`; in `ggml_compute_forward_mul_mat` do the same (`src0_data_local`) and feed it into both `llamafile_sgemm` call sites. `mirror_to_node1` is a safe no-op for pointers outside Buffer A, so non-replicated tensors pass through unchanged. |

### Validation Results (Phase 5, 8B-Q4 test model)
```
llama-cli --no-mmap -p "The capital of France is" -n 32 --single-turn
  → Output: "Paris" (correct)
  → Prompt: 268.7 t/s     (vs 223 Phase 3  = 1.21×;  vs 160 vanilla = 1.68×)
  → Generation: 104.5 t/s (vs  54 Phase 3  = 1.93×;  vs  28 vanilla = 3.73×)

llama-bench -t 192 --mmap 0 -p 512 -n 2048 -r 2
  → pp512:  363.14 ± 2.73 t/s
  → tg2048:  47.04 ± 0.49 t/s (vs 31.85 Phase 3 = 1.48×)

numastat during tg2048 run:
  Private: Node 0 = 8198 MB, Node 1 = 8203 MB (balanced; both buffers fully resident)
  Total  : 16459 MB (= 2 × 8 GiB NUMA buffers + a few tens of MB of activations/KV)
```

Why llama-cli shows a bigger gain than llama-bench tg2048: tg2048 accumulates a large KV cache that still lives entirely on Node 0 (known limitation #4 below), so as sequence length grows, Node 1 workers increasingly pay for remote KV reads. Short generations (tg≤128 or llama-cli with -n 32) are closer to the pure-weight regime where Phase 5 shines.

## State of play right now — Phase 6 Discovery

Just ran grep to find existing thread infrastructure. Key findings:

**llama.cpp ALREADY has most of what we need:**

```
ggml/src/ggml-cpu/ggml-cpu.c:
  line 535:  struct ggml_threadpool { ... }
  line 554:  int n_threads
  line 695:  CPU_SET(core_id, &cpuset);           ← pinning exists!
  line 698:  pthread_setaffinity_np(...)
  line 2241: static void set_numa_thread_affinity(int thread_n)
  line 2711: CPU_SET(i, &cpuset);                  ← per-worker pinning
  line 2721: pthread_setaffinity_np(pthread_self(), ...)
  line 2770: ggml_thread_cpumask_is_valid(...)
  line 2777: ggml_thread_cpumask_next(...)
  line 3101: set_numa_thread_affinity(state->ith)  ← called in worker
  line 3333: ggml_thread_cpumask_next(tpp->cpumask, workers[j].cpumask, tpp->strict_cpu, ...)
```

**NUMA strategies in `ggml-cpu.h`:**
```c
GGML_NUMA_STRATEGY_DISABLED   = 0,
GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
GGML_NUMA_STRATEGY_ISOLATE    = 2,
GGML_NUMA_STRATEGY_NUMACTL    = 3,
GGML_NUMA_STRATEGY_MIRROR     = 4,  ← interesting, unclear if implemented
```

**CLI:** `common/arg.cpp:2246` already handles `--numa distribute|isolate|numactl`.

## Next-session TODO (Phase 6 — deeper perf validation; and possible Phase 4)

### Phase 6 checklist

1. **`perf stat -e LLC-loads,LLC-load-misses,instructions,cycles`** on a long bench run with and without `GGML_NUMA_REPLICATE`. Spec target is 35% reduction in L3 miss rate vs vanilla. We already see the downstream effect in tokens/sec but the miss-rate number is what implement.md asks for.
2. **`perf record --call-graph fp` + `perf report`** on `__lll_lock_wait` specifically. Implement.md targets near-zero lock contention; our strict per-core pinning should get us most of the way, but confirm.
3. **`numastat -p` snapshot during pp (prompt eval)** vs **during tg (generation)** — expect Node 1 `Private` stays high in both, no spill to other nodes.
4. **Bigger model.** The 8B-Q4 test model is weight-bound but the KV cache is small. The real Llama-3.3-70B-Q8 (~70 GB) is where Phase 5's gain should scale further; set `GGML_NUMA_BUFFER_GB=72` and run once to confirm nothing blows up at that size.
5. **Regression baseline.** Run the 8B-Q4 suite with `GGML_NUMA_REPLICATE=0` on this same binary to confirm our code paths are no-ops when the env var is off (i.e., we didn't silently regress the vanilla path).

### Phase 4 (lock-free work-stealing queues) — status
- Robin's earlier code already added a `ggml_task_deque` field to each worker and an `ggml_deque_init` call (called during threadpool creation — Phase 3 wires this).
- Not yet connected to actual work dispatch. `ggml_compute_forward_mul_mat` still uses the single shared atomic `current_chunk` counter (line ~1584 in `ggml-cpu.c`).
- Phase 4 plan: each worker owns a local deque of chunk indices; idle workers steal from neighbors on the same NUMA node before crossing the interconnect. This mostly helps when chunk granularity is uneven.
- Given Phase 5 already hits the 2× headline target for weight-bound inference, Phase 4 may be a smaller marginal win than Phase 6 cleanup. Decide empirically.

### Non-blocking cleanup (anytime)
- `--no-mmap` requirement: intercept `buffer_from_host_ptr` path so mmap + NUMA coexist.
- `CPU_REPACK` disabled: teach `repack.cpp` to use `ggml_numa_alloc_node0` so AVX512 weight-repacking can come back.
- Nested-function warning `ggml_numa_alloc_on_node` inside `ggml_numa_init` — appears dead; delete after confirming.
- Missing prototype for `ggml_pin_thread_to_core`.

## Critical Known Issues / TODOs

1. **`--no-mmap` requirement** — need to intercept `buffer_from_host_ptr` path so mmap works with NUMA too (future phase, not blocking)
2. **`CPU_REPACK` disabled** — need to make `repack.cpp` NUMA-aware to re-enable the AVX512 weight-repack optimization (future phase)
3. ~~**Node 1 buffer still unused**~~ **FIXED in Phase 5** — `ggml_compute_forward_mul_mat` now swaps Node-1 worker reads to Buffer B via `ggml_numa_mirror_to_node1()`
4. **Context / KV / compute buffers still on Node 0** — only weights are replicated. For long-context tg this becomes the dominant remote-read source. Replicating the KV cache (or splitting per layer across nodes) is the obvious next-level optimization, beyond the implement.md spec.

## Key API Surface We've Built

```c
// ggml-numa-alloc.h
int    ggml_numa_alloc_init(size_t size);
void * ggml_numa_alloc_node0(size_t nbytes, size_t align);
void * ggml_numa_mirror_to_node1(const void * ptr_in_buffer_a);  // ← Phase 5 uses this
void * ggml_numa_mirror_to_node0(const void * ptr_in_buffer_b);
void   ggml_numa_replicate_to_node1(void);
bool   ggml_numa_alloc_is_ready(void);

// numa-buffer.h
ggml_backend_buffer_type_t ggml_backend_cpu_numa_buffer_type(void);
bool ggml_backend_cpu_numa_is_enabled(void);
void ggml_backend_cpu_numa_finalize_load(void);
```

## Tone/Working-Style Notes for Next Session

- User runs terminal commands themselves and pastes output (tool's remote terminal doesn't capture stdout in this env)
- User is technically skilled — prefers concise explanations with rationale over hand-holding
- Edits via tool often fail on `llama-model.cpp` (huge file, possibly stale index) — be prepared to fall back to "here's what to paste manually" instructions
- Verify each sub-step with `numastat` / memory-breakdown output before proceeding
- Commit after each phase completes cleanly

## First message for next session should be:

> "Resuming NUMA-aware llama.cpp work. Phase 2 (memory replication) complete and validated. Starting Phase 3 (thread pinning). Please run the 4 commands in 'Step 1: Read what exists' from the handoff doc and paste output — I need to see the existing `ggml_threadpool` struct, `set_numa_thread_affinity`, the per-worker pinning code, and threadpool creation logic before deciding how to integrate 192-thread strict pinning."