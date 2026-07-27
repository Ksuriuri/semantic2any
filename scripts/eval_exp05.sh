#!/usr/bin/env bash
# Eval driver for exp05 = s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x
# Steps: 10000 20000 30000 40000 50000  |  GPUs 0-4
set -euo pipefail

ROOT=/mnt/data_sdd/hhy/noiz-tts/semantic2any
TASK=task12_vctk10pct_min6_prompt3p01
INPUT_DIR=$ROOT/outputs/$TASK/input_full_wav_min6
REF_NONE=$ROOT/outputs/$TASK/reference_tail_style-none
VOCODER=/mnt/data_sdd/hhy/noiz-tts/vae-eval/.hf_cache/models--nvidia--bigvgan_v2_44khz_128band_512x/snapshots/95a9d1dcb12906c03edd938d77b9333d6ded7dfb
CONFIG=$ROOT/configs/s2mel_zipformer_s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x.yaml
EXP_DIR=$ROOT/exp/s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x
LOG_DIR=$ROOT/logs/$TASK
METRIC_ROOT=$ROOT/metrics/$TASK
VAE=/mnt/data_sdd/hhy/noiz-tts/vae-eval
EVAL_DIR=$ROOT/eval
PYTHON=$ROOT/.venv/bin/python
VAE_PYTHON=$VAE/.venv-bigvgan/bin/python

STEPS=(10000 20000 30000 40000 50000)
GPUS=(0 1 2 3 4)

mkdir -p "$LOG_DIR"
DRIVER_LOG=$LOG_DIR/exp05_eval_driver.log
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "START_EXP05 $(date -Is)"

# === PHASE 1: INFERENCE (parallel on GPUs 0-4) ===
echo "=== PHASE1_INFER_START $(date -Is) ==="
pids=()
for idx in "${!STEPS[@]}"; do
  STEP=${STEPS[$idx]}
  GPU=${GPUS[$idx]}
  RUN=exp05_step${STEP}
  GEN_DIR=$ROOT/outputs/$TASK/generated/$RUN
  METRIC_DIR=$METRIC_ROOT/$RUN
  mkdir -p "$GEN_DIR" "$METRIC_DIR"
  echo "INFER_START $(date -Is) run=$RUN gpu=$GPU"
  (
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 \
      "$PYTHON" "$ROOT/scripts/infer_s2mel_zipformer.py" \
        --config "$CONFIG" \
        --checkpoint "$EXP_DIR/s2mel_step${STEP}.pth" \
        --input "$INPUT_DIR" \
        --output-dir "$GEN_DIR" \
        --model-dir "$ROOT/checkpoints/feature-extractors" \
        --vocoder-model "$VOCODER" \
        --device cuda \
        --dtype float16 \
        --prompt-seconds 3.01 \
        --temperature 0.7 \
        --inference-steps 25 \
        --inference-cfg-rate 0.0 \
        --style-mode none \
        --seed 1234 \
      > "$LOG_DIR/${RUN}.infer.log" 2>&1
    echo "INFER_DONE $(date -Is) run=$RUN count=$(find "$GEN_DIR" -maxdepth 1 -name '*.wav' | wc -l)"
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "=== PHASE1_INFER_ALL_DONE $(date -Is) ==="

# === PHASE 2: PAIRED METRICS (parallel, CPU) ===
echo "=== PHASE2_PAIRED_START $(date -Is) ==="
pids=()
for STEP in "${STEPS[@]}"; do
  RUN=exp05_step${STEP}
  GEN_DIR=$ROOT/outputs/$TASK/generated/$RUN
  METRIC_DIR=$METRIC_ROOT/$RUN
  (
    "$PYTHON" "$EVAL_DIR/paired_metrics.py" \
      --reference-dir "$REF_NONE" \
      --generated-dir "$GEN_DIR" \
      --out-json "$METRIC_DIR/paired_metrics.json" \
      --sample-rate 16000 \
      > "$LOG_DIR/${RUN}.paired.log" 2>&1
    echo "PAIRED_DONE $(date -Is) run=$RUN"
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "=== PHASE2_PAIRED_ALL_DONE $(date -Is) ==="

# === PHASE 3: AUDIOLDM METRICS (parallel, GPUs 0-4) ===
echo "=== PHASE3_AUDIOLDM_START $(date -Is) ==="
pids=()
for idx in "${!STEPS[@]}"; do
  STEP=${STEPS[$idx]}
  GPU=${GPUS[$idx]}
  RUN=exp05_step${STEP}
  GEN_DIR=$ROOT/outputs/$TASK/generated/$RUN
  METRIC_DIR=$METRIC_ROOT/$RUN
  (
    CUDA_VISIBLE_DEVICES=$GPU \
      "$VAE_PYTHON" "$VAE/eval_tools/run_audioldm_metrics.py" \
        --reference-dir "$REF_NONE" \
        --generated-dir "$GEN_DIR" \
        --out-json "$METRIC_DIR/audioldm_metrics.json" \
        --sample-rate 16000 \
        --device cuda:0 \
      > "$LOG_DIR/${RUN}.audioldm.log" 2>&1
    echo "AUDIOLDM_DONE $(date -Is) run=$RUN"
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "=== PHASE3_AUDIOLDM_ALL_DONE $(date -Is) ==="

# === PHASE 4: SPEAKER SIMILARITY (parallel, GPUs 0-4) ===
echo "=== PHASE4_SPEAKER_START $(date -Is) ==="
pids=()
for idx in "${!STEPS[@]}"; do
  STEP=${STEPS[$idx]}
  GPU=${GPUS[$idx]}
  RUN=exp05_step${STEP}
  GEN_DIR=$ROOT/outputs/$TASK/generated/$RUN
  METRIC_DIR=$METRIC_ROOT/$RUN
  (
    CUDA_VISIBLE_DEVICES=$GPU \
      "$VAE_PYTHON" "$VAE/eval_tools/run_seed_speaker_similarity.py" \
        --reference-dir "$REF_NONE" \
        --generated-dir "$GEN_DIR" \
        --out-json "$METRIC_DIR/speaker_similarity.json" \
        --checkpoint "$VAE/checkpoints/seed_tts/wavlm_large_finetune.pth" \
        --seed-root "$VAE/external/seed-tts-eval" \
        --device cuda:0 \
      > "$LOG_DIR/${RUN}.speaker.log" 2>&1
    echo "SPEAKER_DONE $(date -Is) run=$RUN"
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "=== PHASE4_SPEAKER_ALL_DONE $(date -Is) ==="

# === PHASE 5: SUMMARIZE ===
echo "=== PHASE5_SUMMARIZE $(date -Is) ==="
"$PYTHON" "$EVAL_DIR/summarize_eval.py" \
  --metric-root "$METRIC_ROOT" \
  --out-tsv "$METRIC_ROOT/summary_sheet_exp05_all.tsv" \
  --out-json "$METRIC_ROOT/summary_exp05_all.json"
echo "SUMMARIZE_DONE $(date -Is)"

# === PHASE 6: PLOT ===
echo "=== PHASE6_PLOT $(date -Is) ==="
PLOT_DIR=$METRIC_ROOT/plots_exp05
mkdir -p "$PLOT_DIR"
"$PYTHON" "$EVAL_DIR/plot_metrics.py" \
  --summary-json "$METRIC_ROOT/summary_exp05_all.json" \
  --experiments exp01 exp02 exp03 exp04 exp05 exp06 \
  --out-dir "$PLOT_DIR"
ls "$PLOT_DIR"/*.png 2>/dev/null | wc -l
tar czf "$METRIC_ROOT/task12_eval_exp05_plots.tar.gz" -C "$METRIC_ROOT" plots_exp05
echo "PLOT_DONE $(date -Is)"

echo "=== ALL_DONE $(date -Is) ==="
