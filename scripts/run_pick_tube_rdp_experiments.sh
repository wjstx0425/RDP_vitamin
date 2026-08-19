#!/usr/bin/env bash
set -euo pipefail

# Sequential single-GPU experiment launcher:
#   1. Re-train the current 02/04/06 PCA30 LDP from scratch for 20 epochs.
#   2. Train full AT + LDP RDP models on 02/04/05/06 for 20 epochs each
#      with total tactile dimensions 30, 16, and 60.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RDP_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${RDP_DIR}"

STAGE=${1:-help}
GPU_ID=${GPU_ID:-0}
PYTHON_BIN=${PYTHON_BIN:-${RDP_DIR}/.venv/bin/python}
LOGGING_MODE=${LOGGING_MODE:-online}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
AT_BATCH=${AT_BATCH:-64}
LDP_BATCH=${LDP_BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
EXPERIMENT_ID=${EXPERIMENT_ID:-$(date +%Y%m%d_%H%M%S)}
DRY_RUN=${DRY_RUN:-0}
FORCE_PREPARE=${FORCE_PREPARE:-0}

# The four smallest local datasets. Override with a whitespace-separated list,
# for example: DATASETS="pick_tube_01 pick_tube_02 pick_tube_04 pick_tube_06".
DATASETS=${DATASETS:-"pick_tube_02 pick_tube_04 pick_tube_05 pick_tube_06"}
read -r -a DATASET_LIST <<< "${DATASETS}"
FOUR_DATASET_TAG=${FOUR_DATASET_TAG:-02_04_05_06}

LEROBOT_ROOT=${LEROBOT_ROOT:-/home/hillbot/datasets}
TACTILE_CACHE_ROOT=${TACTILE_CACHE_ROOT:-${RDP_DIR}/data/tactile_embeddings_encoder0809}
PCA_ROOT=${PCA_ROOT:-${RDP_DIR}/data/PCA_Transform_PickTube}
FOUR_OUTPUT_ROOT=${FOUR_OUTPUT_ROOT:-${RDP_DIR}/data/outputs/pick_tube_${FOUR_DATASET_TAG}}

CURRENT_DATASET_PATH=${CURRENT_DATASET_PATH:-${RDP_DIR}/data/pick_tube_02_04_06_pca30_armwise_rdp_zarr}
CURRENT_OUTPUT_ROOT=${CURRENT_OUTPUT_ROOT:-${RDP_DIR}/data/outputs/pick_tube_02_04_06}
CURRENT_AT_CKPT=${CURRENT_AT_CKPT:-${CURRENT_OUTPUT_ROOT}/at_pca30_armwise_020406_v1/checkpoints/latest.ckpt}

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_pick_tube_rdp_experiments.sh <stage>

Stages:
  current-ldp20  Re-train the current 02/04/06 PCA30 LDP for 20 epochs.
  prepare4       Fit PCA and convert 02/04/05/06 for dimensions 30, 16, 60.
  four30         Prepare and train four-dataset PCA30 AT+LDP for 20 epochs.
  four16         Prepare and train four-dataset PCA16 AT+LDP for 20 epochs.
  four60         Prepare and train four-dataset PCA60 AT+LDP for 20 epochs.
  all            Run all four requested training experiments sequentially.

Common overrides:
  GPU_ID=0 LOGGING_MODE=online MIXED_PRECISION=bf16
  AT_BATCH=64 LDP_BATCH=64 NUM_WORKERS=8
  DATASETS="pick_tube_02 pick_tube_04 pick_tube_05 pick_tube_06"
  FOUR_DATASET_TAG=02_04_05_06
  EXPERIMENT_ID=my_run
  FORCE_PREPARE=1   # refit PCA and replace existing matching Zarr targets
  DRY_RUN=1         # print training commands; data preparation is also skipped

The four-dataset runs train AT for 20 epochs and then LDP for 20 epochs.
Every run uses RESUME=false and a unique EXPERIMENT_ID output directory.
USAGE
}

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

check_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
}

check_four_dataset_inputs() {
  if [[ ${#DATASET_LIST[@]} -ne 4 ]]; then
    echo "DATASETS must contain exactly four dataset names, got: ${DATASETS}" >&2
    exit 2
  fi
  for dataset in "${DATASET_LIST[@]}"; do
    if [[ ! -f "${LEROBOT_ROOT}/${dataset}/meta/info.json" ]]; then
      echo "LeRobot dataset not found: ${LEROBOT_ROOT}/${dataset}" >&2
      exit 1
    fi
    if [[ ! -f "${TACTILE_CACHE_ROOT}/KaiyueChen/${dataset}/embeddings.npy" ]]; then
      echo "Tactile cache not found for ${dataset}" >&2
      exit 1
    fi
  done
}

pca_path() {
  local tactile_dim=$1
  printf '%s/tactile_pca_armwise_%s_2x%d.npz' \
    "${PCA_ROOT}" "${FOUR_DATASET_TAG}" "$((tactile_dim / 2))"
}

dataset_path() {
  local tactile_dim=$1
  printf '%s/data/pick_tube_%s_pca%d_armwise_rdp_zarr' \
    "${RDP_DIR}" "${FOUR_DATASET_TAG}" "${tactile_dim}"
}

prepare_dimension() {
  local tactile_dim=$1
  local components_per_arm=$((tactile_dim / 2))
  local pca_file
  local zarr_dir
  pca_file=$(pca_path "${tactile_dim}")
  zarr_dir=$(dataset_path "${tactile_dim}")

  if (( tactile_dim < 2 || tactile_dim % 2 != 0 )); then
    echo "Total tactile dimension must be a positive even number: ${tactile_dim}" >&2
    exit 2
  fi

  if [[ "${FORCE_PREPARE}" == "1" || ! -f "${pca_file}" ]]; then
    run "${PYTHON_BIN}" fit_pick_tube_tactile_pca.py \
      --tactile-cache-root "${TACTILE_CACHE_ROOT}" \
      --output "${pca_file}" \
      --components-per-arm "${components_per_arm}" \
      --datasets "${DATASET_LIST[@]}"
  else
    echo "Using existing PCA: ${pca_file}"
  fi

  if [[ "${FORCE_PREPARE}" == "1" || ! -d "${zarr_dir}/replay_buffer.zarr" ]]; then
    local convert_args=(
      "${PYTHON_BIN}" convert_pick_tube_lerobot_to_rdp_zarr.py
      --dataset-root "${LEROBOT_ROOT}"
      --tactile-cache-root "${TACTILE_CACHE_ROOT}"
      --output-dir "${zarr_dir}"
      --tactile-pca-path "${pca_file}"
      --datasets "${DATASET_LIST[@]}"
    )
    if [[ "${FORCE_PREPARE}" == "1" ]]; then
      convert_args+=(--overwrite)
    fi
    run "${convert_args[@]}"
  else
    echo "Using existing dataset: ${zarr_dir}"
  fi

  if [[ "${DRY_RUN}" != "1" ]]; then
    run env \
      "PYTHON_BIN=${PYTHON_BIN}" \
      "DATASET_PATH=${zarr_dir}" \
      bash scripts/setup_pick_tube_data.sh validate
  fi
}

train_current_ldp20() {
  if [[ "${DRY_RUN}" != "1" ]]; then
    [[ -d "${CURRENT_DATASET_PATH}/replay_buffer.zarr" ]] || {
      echo "Current dataset not found: ${CURRENT_DATASET_PATH}" >&2
      exit 1
    }
    [[ -f "${CURRENT_AT_CKPT}" ]] || {
      echo "Current AT checkpoint not found: ${CURRENT_AT_CKPT}" >&2
      exit 1
    }
  fi

  run env \
    "AT_CKPT=${CURRENT_AT_CKPT}" \
    "DATASET_PATH=${CURRENT_DATASET_PATH}" \
    "OUTPUT_ROOT=${CURRENT_OUTPUT_ROOT}" \
    "RUN_ID=pca30_armwise_020406_ldp20_${EXPERIMENT_ID}" \
    "GPU_ID=${GPU_ID}" \
    "LOGGING_MODE=${LOGGING_MODE}" \
    "MIXED_PRECISION=${MIXED_PRECISION}" \
    "TACTILE_DIM=30" \
    "LDP_EPOCHS=20" \
    "LDP_BATCH=${LDP_BATCH}" \
    "NUM_WORKERS=${NUM_WORKERS}" \
    "LDP_CHECKPOINT_EVERY=1" \
    "RESUME=false" \
    "VALIDATE_DATASET=0" \
    "DRY_RUN=${DRY_RUN}" \
    bash scripts/train_pick_tube_single_gpu.sh ldp
}

train_four_dimension() {
  local tactile_dim=$1
  local zarr_dir
  zarr_dir=$(dataset_path "${tactile_dim}")
  prepare_dimension "${tactile_dim}"

  run env \
    "DATASET_PATH=${zarr_dir}" \
    "OUTPUT_ROOT=${FOUR_OUTPUT_ROOT}" \
    "RUN_ID=pca${tactile_dim}_armwise_${FOUR_DATASET_TAG}_at20_ldp20_${EXPERIMENT_ID}" \
    "GPU_ID=${GPU_ID}" \
    "LOGGING_MODE=${LOGGING_MODE}" \
    "MIXED_PRECISION=${MIXED_PRECISION}" \
    "TACTILE_DIM=${tactile_dim}" \
    "AT_EPOCHS=20" \
    "LDP_EPOCHS=20" \
    "AT_BATCH=${AT_BATCH}" \
    "LDP_BATCH=${LDP_BATCH}" \
    "NUM_WORKERS=${NUM_WORKERS}" \
    "AT_CHECKPOINT_EVERY=1" \
    "LDP_CHECKPOINT_EVERY=1" \
    "RESUME=false" \
    "VALIDATE_DATASET=0" \
    "DRY_RUN=${DRY_RUN}" \
    bash scripts/train_pick_tube_single_gpu.sh all
}

case "${STAGE}" in
  current-ldp20)
    check_python
    ;;
  prepare4|four30|four16|four60|all)
    check_python
    check_four_dataset_inputs
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

case "${STAGE}" in
  current-ldp20)
    train_current_ldp20
    ;;
  prepare4)
    prepare_dimension 30
    prepare_dimension 16
    prepare_dimension 60
    ;;
  four30)
    train_four_dimension 30
    ;;
  four16)
    train_four_dimension 16
    ;;
  four60)
    train_four_dimension 60
    ;;
  all)
    train_current_ldp20
    train_four_dimension 30
    train_four_dimension 16
    train_four_dimension 60
    ;;
esac

printf '\nRequested experiment stage complete: %s\n' "${STAGE}"
