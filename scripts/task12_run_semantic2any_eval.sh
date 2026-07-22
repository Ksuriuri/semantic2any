#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VAE=${VAE:-/path/to/optional/vae-eval}
RUN=task12_vctk10pct_min6_prompt3p01
OUT_ROOT="$ROOT/outputs/$RUN"
METRIC_ROOT="$ROOT/metrics/$RUN"
LOG_ROOT="$ROOT/logs/$RUN"
INPUT_DIR="$OUT_ROOT/input_full_wav_min6"
REF_STYLE="$OUT_ROOT/reference_tail_style-reference"
REF_NONE="$OUT_ROOT/reference_tail_style-none"
PROMPT_SECONDS=3.01
TEMP=0.7
CFG_RATE=0.0
INFER_STEPS=25
MODEL_DIR=${MODEL_DIR:-$ROOT/checkpoints/feature-extractors}
VOCODER_22=${VOCODER_22:-$ROOT/checkpoints/vocoders/bigvgan_v2_22khz_80band_256x}
VOCODER_44=${VOCODER_44:-$ROOT/checkpoints/vocoders/bigvgan_v2_44khz_128band_512x}

MODE=${1:-full}
mkdir -p "$OUT_ROOT" "$METRIC_ROOT" "$LOG_ROOT"
cd "$ROOT"

run_one() {
  local name=$1
  local cfg=$2
  local ckpt=$3
  local style=$4
  local vocoder=$5
  local gpu=$6
  local input=$INPUT_DIR
  if [[ "$MODE" == "smoke" ]]; then
    input="$OUT_ROOT/input_smoke_one"
    rm -rf "$input"
    mkdir -p "$input"
    local p
    p=$(find "$INPUT_DIR" -maxdepth 1 \( -type l -o -type f \) | sort | sed -n '1p')
    ln -s "$(readlink -f "$p")" "$input/$(basename "$p")"
  fi
  local out_dir="$OUT_ROOT/generated/$name"
  local log="$LOG_ROOT/${name}.infer.log"
  mkdir -p "$out_dir"
  echo "RUN_INFER $(date -Is) name=$name gpu=$gpu style=$style ckpt=$ckpt"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$ROOT/.venv/bin/python" "$ROOT/scripts/infer_s2mel_zipformer.py" \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --input "$input" \
    --output-dir "$out_dir" \
    --model-dir "$MODEL_DIR" \
    --vocoder-model "$vocoder" \
    --device cuda \
    --dtype float16 \
    --prompt-seconds "$PROMPT_SECONDS" \
    --temperature "$TEMP" \
    --inference-steps "$INFER_STEPS" \
    --inference-cfg-rate "$CFG_RATE" \
    --style-mode "$style" \
    --seed 1234 >"$log" 2>&1
  echo "DONE_INFER $(date -Is) name=$name count=$(find "$out_dir" -maxdepth 1 -type f -name '*.wav' | wc -l)"
}

run_metrics_one() {
  local name=$1
  local style=$2
  local gpu=$3
  local gen_dir="$OUT_ROOT/generated/$name"
  local ref_dir=$REF_STYLE
  if [[ "$style" == "none" ]]; then
    ref_dir=$REF_NONE
  fi
  local metric_dir="$METRIC_ROOT/$name"
  mkdir -p "$metric_dir"
  echo "RUN_METRICS $(date -Is) name=$name gpu=$gpu ref=$ref_dir"
  "$ROOT/.venv/bin/python" "$ROOT/scripts/task7_paired_metrics_short.py" \
    --reference-dir "$ref_dir" \
    --generated-dir "$gen_dir" \
    --out-json "$metric_dir/paired_metrics.json" \
    --sample-rate 16000 >"$LOG_ROOT/${name}.paired.log" 2>&1
  PYTHONPATH="$VAE:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="$gpu" "$VAE/.venv-bigvgan/bin/python" "$VAE/eval_tools/run_audioldm_metrics.py" \
    --reference-dir "$ref_dir" \
    --generated-dir "$gen_dir" \
    --out-json "$metric_dir/audioldm_metrics.json" \
    --sample-rate 16000 \
    --device cuda:0 >"$LOG_ROOT/${name}.audioldm.log" 2>&1
  CUDA_VISIBLE_DEVICES="$gpu" "$VAE/.venv-bigvgan/bin/python" "$VAE/eval_tools/run_seed_speaker_similarity.py" \
    --reference-dir "$ref_dir" \
    --generated-dir "$gen_dir" \
    --out-json "$metric_dir/speaker_similarity.json" \
    --checkpoint "$VAE/checkpoints/seed_tts/wavlm_large_finetune.pth" \
    --seed-root "$VAE/external/seed-tts-eval" \
    --device cuda:0 >"$LOG_ROOT/${name}.speaker.log" 2>&1
  echo "DONE_METRICS $(date -Is) name=$name"
}

models=(
  "exp01_step10000|configs/hhy_20260714/exp01_22k_style_zipformer.yaml|exp/exp01_22k_style_zipformer/s2mel_step10000.pth|reference|$VOCODER_22|0"
  "exp01_step20000|configs/hhy_20260714/exp01_22k_style_zipformer.yaml|exp/exp01_22k_style_zipformer/s2mel_step20000.pth|reference|$VOCODER_22|1"
  "exp01_step30000|configs/hhy_20260714/exp01_22k_style_zipformer.yaml|exp/exp01_22k_style_zipformer/s2mel_step30000.pth|reference|$VOCODER_22|2"
  "exp01_step40000|configs/hhy_20260714/exp01_22k_style_zipformer.yaml|exp/exp01_22k_style_zipformer/s2mel_step40000.pth|reference|$VOCODER_22|3"
  "exp02_step10000|configs/hhy_20260714/exp02_44k_style_zipformer.yaml|exp/exp02_44k_style_zipformer/s2mel_step10000.pth|reference|$VOCODER_44|4"
  "exp02_step20000|configs/hhy_20260714/exp02_44k_style_zipformer.yaml|exp/exp02_44k_style_zipformer/s2mel_step20000.pth|reference|$VOCODER_44|5"
  "exp02_step30000|configs/hhy_20260714/exp02_44k_style_zipformer.yaml|exp/exp02_44k_style_zipformer/s2mel_step30000.pth|reference|$VOCODER_44|6"
  "exp02_step40000|configs/hhy_20260714/exp02_44k_style_zipformer.yaml|exp/exp02_44k_style_zipformer/s2mel_step40000.pth|reference|$VOCODER_44|7"
  "exp03_step10000|configs/hhy_20260714/exp03_22k_nostyle_zipformer.yaml|exp/exp03_22k_nostyle_zipformer/s2mel_step10000.pth|none|$VOCODER_22|0"
  "exp03_step20000|configs/hhy_20260714/exp03_22k_nostyle_zipformer.yaml|exp/exp03_22k_nostyle_zipformer/s2mel_step20000.pth|none|$VOCODER_22|1"
  "exp03_step30000|configs/hhy_20260714/exp03_22k_nostyle_zipformer.yaml|exp/exp03_22k_nostyle_zipformer/s2mel_step30000.pth|none|$VOCODER_22|2"
  "exp03_step40000|configs/hhy_20260714/exp03_22k_nostyle_zipformer.yaml|exp/exp03_22k_nostyle_zipformer/s2mel_step40000.pth|none|$VOCODER_22|3"
  "exp04_step10000|configs/hhy_20260714/exp04_22k_style_dit.yaml|exp/exp04_22k_style_dit/s2mel_step10000.pth|reference|$VOCODER_22|4"
)

echo "START $(date -Is) mode=$MODE run=$RUN"

if [[ "$MODE" == "smoke" ]]; then
  smoke_models=("${models[0]}" "${models[4]}" "${models[8]}" "${models[12]}")
  for row in "${smoke_models[@]}"; do
    IFS='|' read -r name cfg ckpt style vocoder gpu <<<"$row"
    run_one "$name" "$cfg" "$ckpt" "$style" "$vocoder" "$gpu"
  done
  echo "SMOKE_DONE $(date -Is)"
  exit 0
fi

if [[ "$MODE" != "metrics" ]]; then
run_row_infer() {
  local row=$1
  IFS='|' read -r name cfg ckpt style vocoder gpu <<<"$row"
  run_one "$name" "$cfg" "$ckpt" "$style" "$vocoder" "$gpu"
}

echo "INFER_WAVE1_START $(date -Is)"
for i in 0 1 2 3 4 5 6 7; do
  run_row_infer "${models[$i]}" &
done
wait
echo "INFER_WAVE1_DONE $(date -Is)"

echo "INFER_WAVE2_START $(date -Is)"
for i in 8 9 10 11 12; do
  run_row_infer "${models[$i]}" &
done
wait
echo "INFER_WAVE2_DONE $(date -Is)"
fi

for row in "${models[@]}"; do
  IFS='|' read -r name cfg ckpt style vocoder gpu <<<"$row"
  run_metrics_one "$name" "$style" "$gpu"
done

"$ROOT/.venv/bin/python" "$ROOT/scripts/task12_summarize_eval.py" \
  --metric-root "$METRIC_ROOT" \
  --out-tsv "$METRIC_ROOT/summary.tsv" \
  --out-json "$METRIC_ROOT/summary.json"

echo "ALL_DONE $(date -Is)"
