#!/bin/bash
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
[[ $# == 3 ]] || { echo 'Usage: sudo bash convert.sh INPUT.onnx DATASET_DIRECTORY NEW_OUTPUT.nb' >&2; exit 1; }
[[ $(uname -m) == x86_64 ]] || { echo 'Requires AMD64/x86_64.' >&2; exit 1; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "$task_root/converter/.venv/bin/python" "$task_root/converter/convert_nb_docker.py" "$@"
