#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec /home/z017/research/curobo_env/bin/python -m hnuter_dit.training "$@"
