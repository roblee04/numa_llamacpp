#!/bin/bash
# A/B bench runner: captures llama-bench stdout + numastat snapshot mid-run.
#
# Usage: run_bench.sh <label> <model> <buffer_gb> <numa_on:0|1> <bench_args...>
# Example:
#   run_bench.sh 8b-numa  ../Meta-Llama-3-8B-Instruct-Q4_K_M.gguf      8  1  --mmap 0 -t 192 -p 512 -n 128 -r 3
#   run_bench.sh 70b-van  ../Llama-3.3-70B-Instruct-Q8_0-00001-of-00002.gguf 72 0  --mmap 0 -t 192 -p 512 -n 128 -r 2

set -euo pipefail

# Load runtime modules for libstdc++ and libnuma
source /etc/profile.d/modules.sh 2>/dev/null || true
module load GCC/14.3.0 2>/dev/null || true
module load numactl/2.0.19-GCCcore-14.3.0 2>/dev/null || true

LABEL="$1"; shift
MODEL="$1"; shift
BUFFER_GB="$1"; shift
NUMA_ON="$1"; shift

BIN=/WAVE/archive/users/rblee/numa_llamacpp/amd_epyc/llama.cpp/build/bin/llama-bench
OUT_DIR=/WAVE/archive/users/rblee/numa_llamacpp/metrics
LOG="$OUT_DIR/$LABEL.log"
NSTAT="$OUT_DIR/$LABEL.numastat"

export GGML_NUMA_REPLICATE="$NUMA_ON"
export GGML_NUMA_BUFFER_GB="$BUFFER_GB"

echo "==== $LABEL ====" | tee "$LOG"
echo "date:              $(date)" | tee -a "$LOG"
echo "GGML_NUMA_REPLICATE=$GGML_NUMA_REPLICATE" | tee -a "$LOG"
echo "GGML_NUMA_BUFFER_GB=$GGML_NUMA_BUFFER_GB" | tee -a "$LOG"
echo "model=$MODEL" | tee -a "$LOG"
echo "args: $*" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Launch bench in background; snapshot numastat after a few seconds.
"$BIN" -m "$MODEL" "$@" >> "$LOG" 2>&1 &
BENCH_PID=$!

# Wait for process to ramp up (load weights), then snapshot
sleep 20
if kill -0 "$BENCH_PID" 2>/dev/null; then
    echo "==== numastat snapshot at t+20s ====" > "$NSTAT"
    numastat -p "$BENCH_PID" >> "$NSTAT" 2>&1 || echo "numastat failed (process may have exited)" >> "$NSTAT"
    echo "" >> "$NSTAT"
    echo "==== /proc/$BENCH_PID/status NUMA-relevant ====" >> "$NSTAT"
    grep -E 'Cpus_allowed_list|Mems_allowed_list|VmRSS|VmData' /proc/"$BENCH_PID"/status >> "$NSTAT" 2>&1 || true
fi

wait "$BENCH_PID"
BENCH_EXIT=$?

echo "" | tee -a "$LOG"
echo "exit_code=$BENCH_EXIT" | tee -a "$LOG"
exit $BENCH_EXIT
