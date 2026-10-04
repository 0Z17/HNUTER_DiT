#!/usr/bin/env bash
# Existing isolated nvblox image; does not install packages or start Isaac GUI.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_dir="${HNUTER_SIM_ROOT:-/home/z017/research/isaacsim-industrial-reactor}"
if [[ $# -lt 1 ]]; then
  echo 'Usage: tools/run_task_gpu.sh hnuter_dit.task_training|hnuter_dit.task_evaluation ARGS...' >&2
  exit 2
fi
exec docker run --rm --gpus all --ipc=host \
  -e LD_LIBRARY_PATH=/opt/nvblox-venv/lib/python3.12/site-packages/nvidia/npp/lib \
  -e PYTHONPATH=/dit -e PYTHONDONTWRITEBYTECODE=1 \
  --mount "type=bind,source=$repo_dir,target=/dit" \
  --mount "type=bind,source=$sim_dir,target=/sim,readonly" \
  --workdir /dit --entrypoint /opt/nvblox-venv/bin/python \
  isaaclab-sim:6.0.0-nvblox -m "$@"
