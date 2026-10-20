#!/usr/bin/env bash
set -euo pipefail
CONFIG=${CONFIG:-configs/s2vae_zipformer.yaml}
TRAIN_JSONL=${TRAIN_JSONL:?Set TRAIN_JSONL to a training manifest}
OUTPUT_DIR=${OUTPUT_DIR:-exp/s2vae_zipformer}
exec accelerate launch trainers/train_s2mel_zipformer.py --config "$CONFIG" --train-jsonl "$TRAIN_JSONL" --output-dir "$OUTPUT_DIR" "$@"
