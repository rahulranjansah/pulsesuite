#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
#  Collect PBS benchmark results into a single table
# ──────────────────────────────────────────────────────────────────────
#  Run this after all bench_threads_pbs.sh array jobs complete:
#    ./bench_collect.sh
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "========================================"
echo "  PulseSuite PBS Benchmark Results"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
printf "%-10s  %-12s  %-12s  %-14s\n" "Threads" "Total (s)" "Loop (s)" "Steps/s"
printf "%-10s  %-12s  %-12s  %-14s\n" "--------" "----------" "----------" "----------"

# Sort by thread count (numeric)
for f in $(ls bench_result_*threads.txt 2>/dev/null | sort -t_ -k2 -n); do
    THR=$(grep -oP 'Threads:\s+\K\d+' "$f" || echo "?")
    TOT=$(grep -oP 'Total:\s+\K[0-9.]+' "$f" || echo "N/A")
    LOOP=$(grep -oP 'Loop:\s+\K[0-9.]+' "$f" || echo "N/A")
    SPS=$(grep -oP 'Steps/s:\s+\K[0-9.]+' "$f" || echo "N/A")
    printf "%-10s  %-12s  %-12s  %-14s\n" "$THR" "$TOT" "$LOOP" "$SPS"
done

echo "========================================"
