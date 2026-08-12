#!/usr/bin/env bash
set -euo pipefail

RDP_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${RDP_DIR}/.." && pwd)
CONFIG_PATH=${CONFIG_PATH:-${RDP_DIR}/pick_tube_tactile_cache_0809.yaml}
JAX_PYTHON=${JAX_PYTHON:-/home/hillbot/flow_matching/.venv/bin/python}

XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${JAX_PYTHON}" "${RDP_DIR}/precompute_pick_tube_v21_tactile_embeddings.py" \
  --config "${CONFIG_PATH}" \
  "$@"
