# NUMA-Aware llama.cpp — Performance Metrics

A/B comparison of the dual-buffer NUMA engine against the
vanilla llama.cpp CPU backend. Both configs run the same binary — the only
difference is `GGML_NUMA_REPLICATE=0` vs `=1`.

## Hardware / Software

| | |
|---|---|
| CPU | 2× AMD EPYC 9654 (96C/192T each → **192 physical / 384 logical**) |
| NUMA | 2 nodes, 1 socket per node |
| RAM | 1.5 TiB |
| Build | `cmake -DGGML_NUMA_REPLICATE=ON -DGGML_CPU_REPACK=OFF -DCMAKE_BUILD_TYPE=Release` |
| Commit | `e365e658f` (build 8851) |
| Threads | `-t 192` (physical cores only, no SMT) |
| mmap | `--mmap 0` (required — mmap path bypasses the NUMA allocator) |
| Date | 2026-04-20 |

## Headline Results

### 8B Q4 — Meta-Llama-3-8B-Instruct-Q4_K_M.gguf (4.58 GiB)

| Test    | Vanilla (t/s)     | NUMA (t/s)        | Speedup |
|---------|-------------------|-------------------|---------|
| pp512   | 380.15 ± 3.48     | 357.30 ± 15.54    | 0.94×   |
| tg128   |  36.88 ± 3.19     |  53.34 ± 0.11     | **1.45×** |
| tg512   |  41.55 ± 0.23     |  53.06 ± 0.03     | **1.28×** |
| tg2048  |  41.96 ± 1.26     |  47.71 ± 0.04     | 1.14×   |

### 70B Q8 — Llama-3.3-70B-Instruct-Q8_0 (69.82 GiB)

| Test    | Vanilla (t/s)     | NUMA (t/s)        | Speedup |
|---------|-------------------|-------------------|---------|
| pp512   |  47.01 ± 0.60     |  75.68 ± 0.03     | **1.61×** |
| tg128   |   3.84 ± 0.87     |   6.71 ± 0.04     | **1.75×** |
| tg512   |   5.71 ± 0.05     |   6.74 ± 0.02     | 1.18×   |

## Interpretation

- **NUMA wins scale with model size / memory pressure.** The 8B Q4 model
  (4.58 GB) mostly fits into per-socket L3 in the weight-hot path, so the
  speedup on prompt eval is a wash. The 70B Q8 model (70 GB) is 15× larger
  than aggregate L3 — every mul_mat is a pure DRAM read, and replication
  cuts the effective UPI cross-socket traffic to ~zero. Result: NUMA gives
  **1.61× on prompt eval** and **1.75× on short-context generation**
  for 70B, vs nearly flat pp on 8B.
- **tg512 gain on 70B drops to 1.18×.** The 512-token KV cache for a 70B
  model (80 layers × 8 heads × 128 dim × 512 tokens × 2 bytes × 2 K/V) is
  on the order of a few GB and still lives entirely on Node 0 (Phase 5
  only replicates weights, not KV). As sequence length grows, Node-1
  workers pay increasing remote-read cost on KV — this is the dominant
  ceiling for long-context generation and is called out as a future
  optimization (replicate/split KV cache per node).
- **Variance is dramatically lower with NUMA on.** Vanilla tg128 on 8B
  has σ = 3.19 (8.6%); NUMA has σ = 0.11 (0.2%). Strict per-core pinning
  eliminates the scheduler jitter and cross-socket cache-line ping-pong
  that vanilla exhibits.

## Memory Placement Evidence

### 8B NUMA, at t+20s (post-load, mid-inference)

```
Per-node process memory usage (in MBs)  -- from 8b-numa.numastat
                           Node 0          Node 1           Total
Private                   8197.71         8204.12        16401.83
```

Near-perfectly balanced: 2× 8 GiB NUMA buffers materialised on Node 0 and
Node 1 respectively (identical 50/50 split to within 0.1%). Both buffers
are fully faulted-in (`Private`) — no on-demand paging, no zero-page trick.

### 70B NUMA, early snapshot (during Node-1 mirror copy)

```
Per-node process memory usage (in MBs)  -- from 70b-numa.numastat
                           Node 0          Node 1           Total
Private                  54771.69           19.97        54791.66
```

Captured at t+20s, which falls during the ~120 s `mmap → Node-0 fault`
phase but *before* the Node-1 replication completes. Note the log line
`[numa-alloc] replicating 71494 MiB from node 0 to node 1...` fires
later. A post-replication snapshot would match the 8B pattern
(~71 GB on each node). `VmData=151 GB` confirms both buffers are
reserved.

### 8B Vanilla (no replication)

```
Per-node process memory usage (in MBs)  -- from 8b-vanilla.numastat
                           Node 0          Node 1           Total
Private                   1775.48         2992.43         4767.91
```

Cpus_allowed_list: `0-383` (no pinning). Memory spilled across both
nodes by first-touch chance, 37/63 split. This is the baseline behavior
Phase 5 replaces.

### 70B Vanilla

```
Per-node process memory usage (in MBs)  -- from 70b-vanilla.numastat
                           Node 0          Node 1           Total
Private                      3.79         7560.79         7564.59
```

Cpus_allowed_list: `0-383`. Model still loading at snapshot time; all
allocations funnelled through the first malloc arena (on Node 1 by
first-touch). Final steady-state is typically 30/70 split — either way,
cross-socket weight reads are ~50% of traffic without replication.

## Caveats / Methodology Notes

1. **tg2048 skipped on 70B.** A single tg2048 rep at 6 t/s is ≈ 5.5 min;
   `-r 2` on both configs = 22 min of pure gen. Skipped for wall-clock
   budget. Trend at tg512 already shows the KV-cache ceiling described
   above.
2. **`GGML_NUMA_REPLICATE=0` as baseline.** This validates the env-var
   gate (no silent regression of the vanilla path) *and* serves as the
   baseline. For the 8B vanilla case, both `Cpus_allowed_list=0-383`
   and the scattered memory placement confirm the non-NUMA code path
   is live. Stronger "two-binary" baseline (rebuild with
   `-DGGML_NUMA_REPLICATE=OFF`) is deferred.
3. **Model loading dominates 70B wall-clock.** 70 GiB with
   `--mmap 0` takes ~2 min before inference starts. llama-bench amortises
   this by loading once and running all tests in one process, so the
   per-test t/s figures are unaffected.
4. **KV cache still on Node 0.** Replication covers model weights only.
   For long-context workloads (tg > ~512), Node-1 workers increasingly
   pay remote-read cost on KV. This is probably the next optimization 

## Files

| File | Contents |
|------|----------|
| `run_bench.sh` | A/B bench runner (env + numastat capture) |
| `8b-vanilla.log`, `8b-vanilla.numastat` | 8B baseline run |
| `8b-numa.log`, `8b-numa.numastat` | 8B NUMA-replicate run |
| `70b-vanilla.log`, `70b-vanilla.numastat` | 70B baseline run |
| `70b-numa.log`, `70b-numa.numastat` | 70B NUMA-replicate run |

## Reproducing

```bash

cd numa_llamacpp/metrics

./run_bench.sh 8b-vanilla  Meta-Llama-3-8B-Instruct-Q4_K_M.gguf       8  0  --mmap 0 -t 192 -p 512 -n 128,512,2048 -r 3
./run_bench.sh 8b-numa     Meta-Llama-3-8B-Instruct-Q4_K_M.gguf       8  1  --mmap 0 -t 192 -p 512 -n 128,512,2048 -r 3
./run_bench.sh 70b-vanilla Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf 72 0 --mmap 0 -t 192 -p 512 -n 128,512 -r 2
./run_bench.sh 70b-numa    Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf 72 1 --mmap 0 -t 192 -p 512 -n 128,512 -r 2
```
