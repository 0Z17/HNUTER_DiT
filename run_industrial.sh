#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_IMAGE="${ISAACSIM_IMAGE:-isaaclab-sim:6.0.0-nvblox}"
TASK_SIM_ROOT="${HNUTER_SIM_ROOT:-/home/z017/research/isaacsim-industrial-reactor}"
TASK_CLI=()
TASK_HEADLESS=false
TASK_CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sim-root)
      [[ $# -ge 2 ]] || { echo '--sim-root requires a host repository path' >&2; exit 1; }
      TASK_SIM_ROOT="$2"; shift 2 ;;
    --sim-root=*) TASK_SIM_ROOT="${1#*=}"; shift ;;
    --temporal-checkpoint)
      [[ $# -ge 2 ]] || { echo '--temporal-checkpoint requires a host file path' >&2; exit 1; }
      TASK_CHECKPOINT="$2"; shift 2 ;;
    --temporal-checkpoint=*) TASK_CHECKPOINT="${1#*=}"; shift ;;
    --headless) TASK_HEADLESS=true; TASK_CLI+=("$1"); shift ;;
    *) TASK_CLI+=("$1"); shift ;;
  esac
done
[[ -f "$TASK_SIM_ROOT/scene_hnuter.usda" ]] || { echo "Industrial simulation repo not found: $TASK_SIM_ROOT" >&2; exit 1; }
TASK_SIM_ROOT="$(cd -- "$TASK_SIM_ROOT" && pwd)"
TASK_DOCKER=(--rm --gpus all --shm-size=2g --network host
  -e PYTHONUNBUFFERED=1 -e PYTHONDONTWRITEBYTECODE=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y
  -e NVIDIA_DRIVER_CAPABILITIES=all
  --mount "type=bind,source=$TASK_ROOT,target=/dit"
  --mount "type=bind,source=$TASK_SIM_ROOT,target=/sim,readonly"
  --mount type=volume,source=refinery-isaacsim60-cache,target=/root/.cache)
if [[ -n "$TASK_CHECKPOINT" ]]; then
  [[ -f "$TASK_CHECKPOINT" ]] || { echo "Temporal checkpoint not found: $TASK_CHECKPOINT" >&2; exit 1; }
  TASK_CHECKPOINT="$(realpath -- "$TASK_CHECKPOINT")"
  TASK_DOCKER+=(--mount "type=bind,source=$TASK_CHECKPOINT,target=/temporal-checkpoint.pt,readonly")
  TASK_CLI+=(--temporal-checkpoint /temporal-checkpoint.pt)
fi
if [[ "$TASK_HEADLESS" == false ]]; then
  : "${DISPLAY:?Run from the graphical desktop or add --headless}"
  TASK_XAUTH="${XAUTHORITY:-$HOME/.Xauthority}"
  [[ -r "$TASK_XAUTH" ]] || { echo 'XAUTHORITY must name a readable desktop authorization file.' >&2; exit 1; }
  TASK_DOCKER+=(-e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/industrial.xauth
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly
    --mount "type=bind,source=$TASK_XAUTH,target=/tmp/industrial.xauth,readonly")
fi
mkdir -p "$TASK_ROOT/results/industrial"
docker run "${TASK_DOCKER[@]}" --entrypoint /bin/bash "$TASK_IMAGE" \
  -lc 'cd /dit && exec /isaac-sim/python.sh -m hnuter_dit.industrial_runner --sim-root /sim "$@"' \
  industrial-launch "${TASK_CLI[@]}" 2>&1 | tee "$TASK_ROOT/results/industrial/launch_$(date +%Y%m%d_%H%M%S)_$$.log"
