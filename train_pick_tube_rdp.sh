#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
GPU_ID=${GPU_ID:-0}
DATASET_PATH=${DATASET_PATH:-data/pick_tube_01_04_rdp_zarr}
OUTPUT_ROOT=${OUTPUT_ROOT:-data/outputs/pick_tube}
LOGGING_MODE=${LOGGING_MODE:-online}
STAGE=${1:-all}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
AT_OUTPUT_DIR=${AT_OUTPUT_DIR:-${OUTPUT_ROOT}/at_${TIMESTAMP}}
LDP_OUTPUT_DIR=${LDP_OUTPUT_DIR:-${OUTPUT_ROOT}/ldp_${TIMESTAMP}}

if [[ "${STAGE}" == "at" || "${STAGE}" == "all" ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} "${PYTHON_BIN}" train.py \
    --config-name=train_pick_tube_at_workspace \
    task.dataset_path="${DATASET_PATH}" \
    hydra.run.dir="${AT_OUTPUT_DIR}" \
    logging.mode="${LOGGING_MODE}"
fi

AT_LOAD_DIR=${AT_LOAD_DIR:-${AT_OUTPUT_DIR}/checkpoints/latest.ckpt}
if [[ "${STAGE}" == "ldp" || "${STAGE}" == "all" ]]; then
  if [[ ! -f "${AT_LOAD_DIR}" ]]; then
    echo "AT checkpoint not found: ${AT_LOAD_DIR}" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES=${GPU_ID} accelerate launch --mixed_precision bf16 train.py \
    --config-name=train_pick_tube_ldp_workspace \
    task.dataset_path="${DATASET_PATH}" \
    at_load_dir="${AT_LOAD_DIR}" \
    hydra.run.dir="${LDP_OUTPUT_DIR}" \
    logging.mode="${LOGGING_MODE}"
fi
