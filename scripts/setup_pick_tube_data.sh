#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${RDP_DIR}"

COMMAND=${1:-validate}
if [[ $# -gt 0 ]]; then
  shift
fi
PYTHON_BIN=${PYTHON_BIN:-${RDP_DIR}/.venv/bin/python}
HF_BIN=${HF_BIN:-${RDP_DIR}/.venv/bin/hf}
LEROBOT_ROOT=${LEROBOT_ROOT:-/DATA/ljl/substage}
TACTILE_CACHE_ROOT=${TACTILE_CACHE_ROOT:-${RDP_DIR}/data/tactile_embeddings_encoder0809}
DATASET_PATH=${DATASET_PATH:-${RDP_DIR}/data/pick_tube_01_04_rdp_zarr}
SMOKE_DATASET_PATH=${SMOKE_DATASET_PATH:-${RDP_DIR}/data/pick_tube_01_04_smoke_rdp_zarr}
ENCODER_DIR=${ENCODER_DIR:-${RDP_DIR}/data/encoder_ckpt_0809}
JAX_PYTHON=${JAX_PYTHON:-${RDP_DIR}/.venv-jax/bin/python}
OVERWRITE=${OVERWRITE:-0}
SMOKE_EPISODES=${SMOKE_EPISODES:-1}

usage() {
  cat <<USAGE
Usage: $0 <encoder|precompute|validate|convert|smoke>

  encoder    Download the inference-only tactile encoder checkpoint from HF.
  precompute Build tactile embeddings using a separate JAX CUDA environment.
  validate   Validate an existing RDP Zarr and load one AT/LDP batch.
  convert    Convert all LeRobot pick_tube_01..04 episodes, then validate.
  smoke      Convert a small subset to SMOKE_DATASET_PATH, then validate.

Environment variables:
  LEROBOT_ROOT        LeRobot root containing pick_tube_01..04
                      (default: /DATA/ljl/substage)
  TACTILE_CACHE_ROOT  Root containing KaiyueChen/pick_tube_XX/embeddings.npy
  DATASET_PATH        Full RDP Zarr output/input directory
  JAX_PYTHON          Python from a separate CUDA-enabled JAX environment
  ENCODER_DIR         Downloaded encoder checkpoint directory
  OVERWRITE=1         Replace an existing conversion target
USAGE
}

download_encoder() {
  "${HF_BIN}" download KaiyueChen/encoder_ckpt_0809 \
    --include checkpoint.json \
    --include 'params-*.npz' \
    --local-dir "${ENCODER_DIR}"
}

precompute_tactile() {
  if [[ -z "${JAX_PYTHON}" || ! -x "${JAX_PYTHON}" ]]; then
    echo "Set JAX_PYTHON to a CUDA-enabled JAX environment." >&2
    exit 1
  fi
  if [[ ! -f "${ENCODER_DIR}/checkpoint.json" ]]; then
    download_encoder
  fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "${JAX_PYTHON}" precompute_pick_tube_v21_tactile_embeddings.py \
      --dataset-root "${LEROBOT_ROOT}" \
      --cache-root "${TACTILE_CACHE_ROOT}" \
      --encoder-path "${ENCODER_DIR}" \
      "$@"
}

validate_dataset() {
  local dataset_path=$1
  if [[ ! -d "${dataset_path}/replay_buffer.zarr" ]]; then
    echo "RDP Zarr not found: ${dataset_path}/replay_buffer.zarr" >&2
    exit 1
  fi

  "${PYTHON_BIN}" - "${dataset_path}" <<'PY'
from pathlib import Path
import sys
import zarr

path = Path(sys.argv[1]) / "replay_buffer.zarr"
root = zarr.open_group(str(path), mode="r")
expected = {
    "camera1": (224, 224, 3),
    "camera2": (224, 224, 3),
    "observation_state": (20,),
    "tactile_embedding": (2048,),
    "action": (20,),
}
for key, tail in expected.items():
    shape = root["data"][key].shape
    if shape[1:] != tail:
        raise SystemExit(f"{key}: expected [N,{tail}], got {shape}")
lengths = {root["data"][key].shape[0] for key in expected}
if len(lengths) != 1:
    raise SystemExit(f"frame counts disagree: {lengths}")
episode_ends = root["meta"]["episode_ends"][:]
if len(episode_ends) == 0 or int(episode_ends[-1]) != next(iter(lengths)):
    raise SystemExit("episode_ends does not match frame count")
print(f"contract OK: episodes={len(episode_ends)}, frames={int(episode_ends[-1])}")
PY
  "${PYTHON_BIN}" validate_pick_tube_batch.py "${dataset_path}" --mode at --batch-size 2
  "${PYTHON_BIN}" validate_pick_tube_batch.py "${dataset_path}" --mode ldp --batch-size 2
}

convert_dataset() {
  local output_path=$1
  shift
  local args=(
    --dataset-root "${LEROBOT_ROOT}"
    --tactile-cache-root "${TACTILE_CACHE_ROOT}"
    --output-dir "${output_path}"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  args+=("$@")
  "${PYTHON_BIN}" convert_pick_tube_lerobot_to_rdp_zarr.py "${args[@]}"
  validate_dataset "${output_path}"
}

case "${COMMAND}" in
  encoder)
    download_encoder
    ;;
  precompute)
    precompute_tactile "$@"
    ;;
  validate)
    validate_dataset "${DATASET_PATH}"
    ;;
  convert)
    convert_dataset "${DATASET_PATH}" "$@"
    ;;
  smoke)
    convert_dataset "${SMOKE_DATASET_PATH}" --max-episodes-per-dataset "${SMOKE_EPISODES}" "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
