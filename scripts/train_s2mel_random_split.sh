#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered}
CODE_ROOT=${CODE_ROOT:-${DATASET_ROOT}/maskgct-codes}
SPLIT_ROOT=${SPLIT_ROOT:-${CODE_ROOT}/splits/seed1234_valid1000}
TRAIN_JSONL=${TRAIN_JSONL:-${SPLIT_ROOT}/train.jsonl}
VALID_JSONL=${VALID_JSONL:-${SPLIT_ROOT}/valid.jsonl}
CONFIG=${CONFIG:-configs/s2mel_zipformer_s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-exp/s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x}
LOG_FILE=${LOG_FILE:-${OUTPUT_DIR}/train.log}
NUM_PROCESSES=${NUM_PROCESSES:-8}
NUM_MACHINES=${NUM_MACHINES:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29543}

mkdir -p "${OUTPUT_DIR}"

uv run accelerate launch \
  --multi_gpu \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  trainers/train_s2mel_zipformer.py \
  --config "${CONFIG}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --valid-jsonl "${VALID_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@" 2>&1 | tee -a "${LOG_FILE}"
