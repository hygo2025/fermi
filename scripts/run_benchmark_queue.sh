#!/usr/bin/env bash
set -euo pipefail

# Run a queue of models, waiting for any existing run_benchmark.py to finish.
#
# Usage:
#   MODELS="SASRec SRGNN GCSAN" scripts/run_benchmark_queue.sh
#   WAIT_PATTERN="src/run_benchmark.py" POLL_SECONDS=60 MODELS="..." scripts/run_benchmark_queue.sh
#

WAIT_PATTERN="${WAIT_PATTERN:-src/run_benchmark.py}"
POLL_SECONDS="${POLL_SECONDS:-60}"

if [[ -z "${MODELS:-}" ]]; then
  echo "[ERROR] MODELS is required. Example: MODELS=\"SASRec SRGNN\" scripts/run_benchmark_queue.sh"
  exit 1
fi

IFS=' ' read -r -a MODELS_LIST <<< "${MODELS}"

echo "[INFO] Queue size: ${#MODELS_LIST[@]}"
echo "[INFO] Waiting pattern: ${WAIT_PATTERN}"
echo "[INFO] Poll interval: ${POLL_SECONDS}s"

current_model() {
  local line
  line="$(pgrep -fa "${WAIT_PATTERN}" | head -n 1 || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return
  fi
  # Extract model name from: ... run_benchmark.py --model MODEL
  if [[ "${line}" =~ --model[[:space:]]+([^[:space:]]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

for model in "${MODELS_LIST[@]}"; do
  echo ""
  echo "[INFO] Queued model: ${model}"
  while pgrep -f "${WAIT_PATTERN}" >/dev/null 2>&1; do
    running_model="$(current_model)"
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    if [[ -n "${running_model}" ]]; then
      echo "[INFO] ${timestamp} Another run_benchmark.py is running (${running_model})... checking again in ${POLL_SECONDS}s"
    else
      echo "[INFO] ${timestamp} Another run_benchmark.py is running... checking again in ${POLL_SECONDS}s"
    fi
    sleep "${POLL_SECONDS}"
  done
  echo "[INFO] Starting: ${model}"
  MODELS="${model}" make benchmark
done

echo ""
echo "[INFO] Queue complete!"
