#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${HNUTER_DIT_PYTHON:-/home/z017/research/curobo_env/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/hnuter-dit-mpl}"

cd "${repo_dir}"
exec "${python_bin}" -m hnuter_dit.contact_visualization "$@"
