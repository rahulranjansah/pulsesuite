#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
#  PulseSuite Thread Benchmark — PBS Array Job
# ──────────────────────────────────────────────────────────────────────
#  Submits one job per thread count using a PBS array.
#
#  Usage:
#    qsub bench_threads_pbs.sh
#
#  Each array index maps to a thread count:
#    Index 0 → 1 thread
#    Index 1 → 2 threads
#    Index 2 → 4 threads
#    Index 3 → 8 threads
#    Index 4 → 16 threads
#    Index 5 → 32 threads
#    Index 6 → 48 threads
#    Index 7 → 64 threads
#
#  After all jobs finish, collect results:
#    cat bench_result_*.txt
#
#  Adjust ncpus/mem/walltime below if needed for your allocation.
# ──────────────────────────────────────────────────────────────────────

#PBS -N ps_bench
#PBS -A cesm0029
#PBS -l select=1:ncpus=64:mpiprocs=1:mem=235G
#PBS -l walltime=02:00:00
#PBS -q main
#PBS -j oe
#PBS -J 0-7

# ── Thread-count lookup table ─────────────────────────────────────
THREAD_COUNTS=(1 2 4 8 16 32 48 64)
N=${THREAD_COUNTS[$PBS_ARRAY_INDEX]}

# ── Environment ───────────────────────────────────────────────────
export NUMBA_NUM_THREADS=$N
export OMP_NUM_THREADS=$N
export MKL_NUM_THREADS=$N
export PULSESUITE_USE_CUDA=0      # CPU-only benchmark
export PATH="$HOME/.local/bin:$PATH"

# Which test to run
TEST_MODULE="pulsesuite.PSTD3D.sbetestprop"

cd "$PBS_O_WORKDIR/src/pulsesuite/PSTD3D"

echo "========================================"
echo "  PBS Thread Benchmark"
echo "  Array index: $PBS_ARRAY_INDEX"
echo "  Threads:     $N"
echo "  Test:        $TEST_MODULE"
echo "  CUDA:        OFF"
echo "========================================"

uv run python -u -m "$TEST_MODULE"

# ── Extract results from latest run ───────────────────────────────
LATEST="$(readlink -f runs/latest 2>/dev/null || echo '')"
RESULT_FILE="$PBS_O_WORKDIR/bench_result_${N}threads.txt"

if [ -n "$LATEST" ] && [ -f "$LATEST/run_summary.txt" ]; then
    TOTAL=$(grep -oP '(?<=total\s{2,})\d+:\d+:\d+\.\d+\s+\(\K[0-9.]+' "$LATEST/run_summary.txt" || echo "N/A")
    LOOP=$(grep -oP '(?<=timeloop\s{2,})\d+:\d+:\d+\.\d+\s+\(\K[0-9.]+' "$LATEST/run_summary.txt" || echo "N/A")
    STEPS=$(grep -oP 'Steps per second:\s+\K[0-9.]+' "$LATEST/run_summary.txt" || echo "N/A")

    {
        echo "Threads: $N"
        echo "Total:   ${TOTAL} s"
        echo "Loop:    ${LOOP} s"
        echo "Steps/s: ${STEPS}"
        echo "Run dir: $LATEST"
    } > "$RESULT_FILE"

    echo ""
    echo "  Result: Threads=$N  Total=${TOTAL}s  Loop=${LOOP}s  Steps/s=${STEPS}"
    echo "  Saved:  $RESULT_FILE"
else
    echo "WARNING: run_summary.txt not found for $N threads" > "$RESULT_FILE"
fi
