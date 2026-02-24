#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
#  PulseSuite Thread Benchmark (local, no scheduler)
# ──────────────────────────────────────────────────────────────────────
#  Usage:
#    chmod +x bench_threads.sh
#    ./bench_threads.sh                  # default: 1 2 4 8
#    ./bench_threads.sh 1 2 4 8 16 32   # custom thread counts
#
#  Each run creates its own timestamped folder in runs/.
#  Results are collected into bench_results.txt at the end.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Which test to run (change to sbetestpropnov30 for 3D)
TEST_MODULE="${BENCH_TEST:-pulsesuite.PSTD3D.sbetestprop}"

# Force CPU so we are benchmarking threads only, not GPU
export PULSESUITE_USE_CUDA=0

# Thread counts: use arguments if provided, else defaults
if [ $# -gt 0 ]; then
    THREADS=("$@")
else
    THREADS=(1 2 4 8)
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$SCRIPT_DIR/src/pulsesuite/PSTD3D"
RESULTS_FILE="$SCRIPT_DIR/bench_results.txt"

echo "========================================"
echo "  PulseSuite Thread Benchmark"
echo "========================================"
echo "  Test module:  $TEST_MODULE"
echo "  Thread counts: ${THREADS[*]}"
echo "  CUDA:          OFF (CPU only)"
echo "  Working dir:   $RUN_DIR"
echo "========================================"
echo ""

# Header for results file
{
    echo "========================================"
    echo "  PulseSuite Thread Benchmark Results"
    echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Test: $TEST_MODULE"
    echo "========================================"
    printf "%-8s  %-12s  %-12s  %-14s\n" "Threads" "Total (s)" "Loop (s)" "Steps/s"
    printf "%-8s  %-12s  %-12s  %-14s\n" "-------" "----------" "----------" "----------"
} > "$RESULTS_FILE"

cd "$RUN_DIR"

for N in "${THREADS[@]}"; do
    echo "── Running with $N threads ──────────────────────"
    export NUMBA_NUM_THREADS=$N
    export OMP_NUM_THREADS=$N
    export MKL_NUM_THREADS=$N

    uv run python -u -m "$TEST_MODULE" 2>&1 | tee "/tmp/pulsesuite_bench_${N}.log"

    # Find the latest run directory
    LATEST="$(readlink -f runs/latest 2>/dev/null || echo '')"
    if [ -z "$LATEST" ] || [ ! -f "$LATEST/run_summary.txt" ]; then
        echo "  WARNING: Could not find run_summary.txt for $N threads"
        continue
    fi

    # Extract timing from run_summary.txt
    TOTAL=$(grep -oP '(?<=total\s{2,})\d+:\d+:\d+\.\d+\s+\(\K[0-9.]+' "$LATEST/run_summary.txt" || echo "N/A")
    LOOP=$(grep -oP '(?<=timeloop\s{2,})\d+:\d+:\d+\.\d+\s+\(\K[0-9.]+' "$LATEST/run_summary.txt" || echo "N/A")
    STEPS=$(grep -oP 'Steps per second:\s+\K[0-9.]+' "$LATEST/run_summary.txt" || echo "N/A")

    printf "%-8s  %-12s  %-12s  %-14s\n" "$N" "$TOTAL" "$LOOP" "$STEPS" >> "$RESULTS_FILE"

    echo "  Threads=$N  Total=${TOTAL}s  Loop=${LOOP}s  Steps/s=${STEPS}"
    echo ""
done

echo "" >> "$RESULTS_FILE"
echo "======================================" >> "$RESULTS_FILE"

echo ""
echo "========================================"
echo "  BENCHMARK COMPLETE"
echo "========================================"
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Results saved to: $RESULTS_FILE"
