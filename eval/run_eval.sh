#!/usr/bin/env bash
# Run inference and metrics for a single s2mel checkpoint.
#
# Required:
#   RUN_NAME    short label for this run, used as metrics/outputs subdir name
#   CHECKPOINT  path to .pth checkpoint file
#   CONFIG      path to experiment YAML config
#   VOCODER     path to BigVGAN vocoder directory
#   INPUT_REF   root directory prepared by eval/prepare_vctk_refs.py
#
# Optional:
#   STYLE_MODE      reference | none  (default: none)
#   PROMPT_SECONDS  (default: 3.01)
#   TEMPERATURE     (default: 0.7)
#   CFG_RATE        (default: 0.0)
#   INFER_STEPS     (default: 25)
#   SEED            (default: 1234)
#   GPU             CUDA device index (default: 0)
#   MODEL_DIR       feature-extractor checkpoints (default: checkpoints/feature-extractors)
#   VAE             path to vae-eval project; if unset, audioldm + speaker-sim steps are skipped
#   SMOKE           1 = run on a single utterance only (default: 0)
#   MODE            full | infer | metrics  (default: full)
#
# Metrics layout:
#   metrics/<RUN_NAME>/paired_metrics.json      — always produced (SI-SDR, LSD)
#   metrics/<RUN_NAME>/audioldm_metrics.json    — requires VAE
#   metrics/<RUN_NAME>/speaker_similarity.json  — requires VAE
#
# To compare multiple runs, call summarize_eval.py with --metric-root metrics/.
#
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
EVAL_DIR="$ROOT/eval"

RUN_NAME=${RUN_NAME:?'RUN_NAME required, e.g. exp06_step34000_prompt3_cfg0_steps25'}
CHECKPOINT=${CHECKPOINT:?'CHECKPOINT required'}
CONFIG=${CONFIG:?'CONFIG required'}
VOCODER=${VOCODER:?'VOCODER required'}
INPUT_REF=${INPUT_REF:?'INPUT_REF required (output of prepare_vctk_refs.py)'}

STYLE_MODE=${STYLE_MODE:-none}
PROMPT_SECONDS=${PROMPT_SECONDS:-3.01}
TEMPERATURE=${TEMPERATURE:-0.7}
CFG_RATE=${CFG_RATE:-0.0}
INFER_STEPS=${INFER_STEPS:-25}
SEED=${SEED:-1234}
GPU=${GPU:-0}
MODEL_DIR=${MODEL_DIR:-$ROOT/checkpoints/feature-extractors}
VAE=${VAE:-}
SMOKE=${SMOKE:-0}
MODE=${MODE:-full}

GEN_DIR="$ROOT/outputs/$RUN_NAME/generated"
METRIC_DIR="$ROOT/metrics/$RUN_NAME"
LOG_DIR="$ROOT/logs/$RUN_NAME"
INPUT_DIR="$INPUT_REF/input_full_wav"
REF_DIR="$INPUT_REF/reference_tail_style-${STYLE_MODE}"

mkdir -p "$GEN_DIR" "$METRIC_DIR" "$LOG_DIR"

# Smoke: restrict to one utterance
INFER_INPUT="$INPUT_DIR"
if [[ "$SMOKE" == "1" ]]; then
  SMOKE_DIR="$ROOT/outputs/$RUN_NAME/_smoke_input"
  rm -rf "$SMOKE_DIR" && mkdir -p "$SMOKE_DIR"
  find "$INPUT_DIR" -maxdepth 1 \( -type l -o -type f \) | sort | head -1 | \
    xargs -I{} sh -c 'ln -s "$(readlink -f "$1")" "$2/$(basename "$1")"' _ {} "$SMOKE_DIR"
  INFER_INPUT="$SMOKE_DIR"
fi

# Inference
if [[ "$MODE" != "metrics" ]]; then
  echo "INFER_START $(date -Is) run=$RUN_NAME steps=$INFER_STEPS cfg=$CFG_RATE"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "$ROOT/.venv/bin/python" "$ROOT/scripts/infer_s2mel_zipformer.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --input "$INFER_INPUT" \
      --output-dir "$GEN_DIR" \
      --model-dir "$MODEL_DIR" \
      --vocoder-model "$VOCODER" \
      --device cuda \
      --dtype float16 \
      --prompt-seconds "$PROMPT_SECONDS" \
      --temperature "$TEMPERATURE" \
      --inference-steps "$INFER_STEPS" \
      --inference-cfg-rate "$CFG_RATE" \
      --style-mode "$STYLE_MODE" \
      --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/infer.log"
  echo "INFER_DONE $(date -Is) count=$(find "$GEN_DIR" -maxdepth 1 -name '*.wav' | wc -l)"
fi

# Metrics
if [[ "$MODE" != "infer" ]]; then
  echo "PAIRED_METRICS_START $(date -Is)"
  "$ROOT/.venv/bin/python" "$EVAL_DIR/paired_metrics.py" \
    --reference-dir "$REF_DIR" \
    --generated-dir "$GEN_DIR" \
    --out-json "$METRIC_DIR/paired_metrics.json" \
    --sample-rate 16000 \
    2>&1 | tee "$LOG_DIR/paired_metrics.log"
  echo "PAIRED_METRICS_DONE $(date -Is)"

  if [[ -n "$VAE" ]]; then
    echo "AUDIOLDM_START $(date -Is)"
    CUDA_VISIBLE_DEVICES="$GPU" \
      "$VAE/.venv-bigvgan/bin/python" "$VAE/eval_tools/run_audioldm_metrics.py" \
        --reference-dir "$REF_DIR" \
        --generated-dir "$GEN_DIR" \
        --out-json "$METRIC_DIR/audioldm_metrics.json" \
        --sample-rate 16000 \
        --device cuda:0 \
      2>&1 | tee "$LOG_DIR/audioldm_metrics.log"
    echo "AUDIOLDM_DONE $(date -Is)"

    echo "SPEAKER_SIM_START $(date -Is)"
    CUDA_VISIBLE_DEVICES="$GPU" \
      "$VAE/.venv-bigvgan/bin/python" "$VAE/eval_tools/run_seed_speaker_similarity.py" \
        --reference-dir "$REF_DIR" \
        --generated-dir "$GEN_DIR" \
        --out-json "$METRIC_DIR/speaker_similarity.json" \
        --checkpoint "$VAE/checkpoints/seed_tts/wavlm_large_finetune.pth" \
        --seed-root "$VAE/external/seed-tts-eval" \
        --device cuda:0 \
      2>&1 | tee "$LOG_DIR/speaker_similarity.log"
    echo "SPEAKER_SIM_DONE $(date -Is)"
  else
    echo "VAE not set — skipping audioldm and speaker-similarity metrics"
  fi

  echo "ALL_DONE $(date -Is) run=$RUN_NAME"
fi
