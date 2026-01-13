#!/usr/bin/env bash
set -euo pipefail

# Benchmark runner script
# Usage: ./scripts/run_benchmark.sh [MODEL|GROUP]
#   MODEL/GROUP: specific model name, 'neurais', 'baselines', 'factorization', or 'all' (default)

TARGET=${1:-all}

echo "[INFO] Running benchmark for: ${TARGET}"
python src/run_benchmark.py --models "${TARGET}"
echo "[INFO] Benchmark complete for ${TARGET}"
