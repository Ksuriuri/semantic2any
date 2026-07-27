#!/usr/bin/env bash
# Re-run phases 3-6 for exp05 eval with correct PYTHONPATH for audioldm.
set -euo pipefail

ROOT=/mnt/data_sdd/hhy/noiz-tts/semantic2any
TASK=task12_vctk10pct_min6_prompt3p01
REF_NONE=$ROOT/outputs/$TASK/reference_tail_style-none
LOG_DIR=$ROOT/logs/$TASK
METRIC_ROOT=$ROOT/metrics/$TASK
VAE=/mnt/data_sdd/hhy/noiz-tts/vae-eval
PYTHON=$ROOT/.venv/bin/python
VAE_PYTHON=$VAE/.venv-bigvgan/bin/python
# Key fix: PYTHONPATH must include ssr_eval dir and audioldm_eval package dir
export PYTHONPATH="$VAE/external/audioldm_eval:$VAE:${PYTHONPATH:-}"

STEPS=(10000 20000 30000 40000 50000)
GPUS=(0 1 2 3 4)

DRIVER_LOG=$LOG_DIR/exp05_eval_driver.log
exec >> "$DRIVER_LOG" 2>&1

echo "RESUME_PHASES3TO6 $(date -Is)"

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
"$PYTHON" "$ROOT/eval/summarize_eval.py" \
  --metric-root "$METRIC_ROOT" \
  --out-tsv "$METRIC_ROOT/summary_sheet_exp05_all.tsv" \
  --out-json "$METRIC_ROOT/summary_exp05_all.json"
echo "SUMMARIZE_DONE $(date -Is)"

# === PHASE 6: PLOT ===
echo "=== PHASE6_PLOT $(date -Is) ==="
PLOT_DIR=$METRIC_ROOT/plots_exp05
mkdir -p "$PLOT_DIR"
"$PYTHON" "$ROOT/eval/plot_metrics.py" \
  --summary-json "$METRIC_ROOT/summary_exp05_all.json" \
  --experiments exp01 exp02 exp03 exp04 exp05 exp06 \
  --out-dir "$PLOT_DIR"
echo "Plot PNGs: $(ls "$PLOT_DIR"/*.png 2>/dev/null | wc -l)"
tar czf "$METRIC_ROOT/task12_eval_exp05_plots.tar.gz" -C "$METRIC_ROOT" plots_exp05
echo "PLOT_DONE $(date -Is)"

echo "=== ALL_DONE $(date -Is) ==="
