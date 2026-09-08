#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_dir}"
exec /home/z017/research/curobo_env/bin/python -m hnuter_dit.training \
  --model-type dit_cross \
  --output-dir "${repo_dir}/artifacts/base_free_flight_159103traj_dit_canonical_cross_v003" \
  "$@"
