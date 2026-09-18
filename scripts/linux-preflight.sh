#!/usr/bin/env bash
set -euo pipefail

for command in docker nvidia-smi; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing required command: ${command}" >&2
    exit 1
  fi
done

docker info >/dev/null

gpu_rows="$(nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader,nounits)"
gpu_count="$(printf '%s\n' "${gpu_rows}" | sed '/^[[:space:]]*$/d' | wc -l)"
rtx_3090_count="$(printf '%s\n' "${gpu_rows}" | grep -c '3090' || true)"
rtx_4090_count="$(printf '%s\n' "${gpu_rows}" | grep -c '4090' || true)"

printf '%s\n' "${gpu_rows}"
echo "Detected GPUs: total=${gpu_count}, RTX3090=${rtx_3090_count}, RTX4090=${rtx_4090_count}"

if [[ "${rtx_3090_count}" -ne 2 || "${rtx_4090_count}" -ne 2 ]]; then
  echo "Expected exactly two RTX 3090 and two RTX 4090 devices." >&2
  exit 1
fi

docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null
echo "Docker and NVIDIA Container Toolkit preflight passed."
