#!/usr/bin/env bash
# Listening test: exp05 step93000 on valid set same-speaker pairs
# steps=200, cfg=0.7 (matching the prior steps200_cfg0p7 run); serial on 1 GPU to coexist with training
set -euo pipefail

ROOT=/mnt/data_sdd/hhy/noiz-tts/semantic2any
EXP_DIR=$ROOT/exp/s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x
VOCODER=/mnt/data_sdd/hhy/noiz-tts/vae-eval/.hf_cache/models--nvidia--bigvgan_v2_44khz_128band_512x/snapshots/95a9d1dcb12906c03edd938d77b9333d6ded7dfb
CONFIG=$ROOT/configs/s2mel_zipformer_s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x.yaml
VENV=$ROOT/.venv/bin/python
DATE=20260728
STEP=93000
INFER_STEPS=200
CFG=0.7
GPU=1
OUT_ROOT=$ROOT/outputs/exp05_valid_listening_${DATE}_step${STEP}_steps${INFER_STEPS}_cfg0p7

mkdir -p "$OUT_ROOT"

echo "=== Phase 1: Prepare concatenated wavs ==="
PREP_DIR="$OUT_ROOT/prep"
mkdir -p "$PREP_DIR"

declare -a SPEAKERS=( "laion_emolia_ZH_B00031" "laion_emolia_ZH_B00064" "StarRail_fugue_en" )
declare -a A_PATHS=(
  "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/laion_emolia/laion_emolia-033361__laion_emolia__ZH_B00031_S04396_W000072__e63d15e06d0a.flac"
  "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/laion_emolia/laion_emolia-036746__laion_emolia__ZH_B00064_S08181_W000017__47cbf7c293d9.flac"
  "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/StarRail/StarRail-000014__StarRail__en_archive_fugue_9__85ce88f2f16f.flac"
)
declare -a B_PATHS=(
  "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/laion_emolia/laion_emolia-033361__laion_emolia__ZH_B00031_S04396_W000004__f7e7d5e3789d.flac"
  "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/laion_emolia/laion_emolia-036746__laion_emolia__ZH_B00064_S08181_W000021__dbc5ee303904.flac"
  "/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/StarRail/StarRail-000014__StarRail__en_archive_fugue_12__2f327c57e089.flac"
)
declare -a A_DURS=( "17.27" "16.6" "18.61" )

for i in 0 1 2; do
  spk="${SPEAKERS[$i]}"
  a="${A_PATHS[$i]}"
  b="${B_PATHS[$i]}"
  tmp_a="$PREP_DIR/pair${i}_a.wav"
  tmp_b="$PREP_DIR/pair${i}_b.wav"
  concat="$PREP_DIR/pair${i}_concat.wav"
  ffmpeg -y -i "$a" -ar 44100 -ac 1 "$tmp_a" 2>/dev/null
  ffmpeg -y -i "$b" -ar 44100 -ac 1 "$tmp_b" 2>/dev/null
  ffmpeg -y -i "$tmp_a" -i "$tmp_b" -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1" "$concat" 2>/dev/null
  echo "pair${i} ($spk) prepared"
done

echo ""
echo "=== Phase 2: Inference step${STEP} (serial on GPU ${GPU}, steps=${INFER_STEPS}, cfg=${CFG}) ==="
ckpt="$EXP_DIR/s2mel_step${STEP}.pth"
gen_dir="$OUT_ROOT/generated_step${STEP}"
mkdir -p "$gen_dir"
for i in 0 1 2; do
  dur_a="${A_DURS[$i]}"
  concat="$PREP_DIR/pair${i}_concat.wav"
  echo "--- pair${i} ---"
  CUDA_VISIBLE_DEVICES=$GPU $VENV $ROOT/scripts/infer_s2mel_zipformer.py \
    --config "$CONFIG" \
    --checkpoint "$ckpt" \
    --input "$concat" \
    --output-dir "$gen_dir" \
    --vocoder-model "$VOCODER" \
    --model-dir "$ROOT/checkpoints/feature-extractors" \
    --prompt-seconds "$dur_a" \
    --inference-steps "$INFER_STEPS" \
    --inference-cfg-rate "$CFG" \
    --temperature 0.7 \
    --style-mode none \
    --seed 1234 \
    --dtype float16 \
    2>"$OUT_ROOT/infer_step${STEP}_pair${i}.log"
done
echo "step${STEP} done: $(ls $gen_dir/*.wav 2>/dev/null | wc -l) files"

echo ""
echo "=== Phase 3: Package ==="
PKG_DIR="$OUT_ROOT/exp05_valid_pair_listening_${DATE}_step${STEP}_steps${INFER_STEPS}_cfg0p7"
mkdir -p "$PKG_DIR"
for i in 0 1 2; do
  spk="${SPEAKERS[$i]}"
  sd="$PKG_DIR/sample${i}_${spk}"
  mkdir -p "$sd"
  cp "$PREP_DIR/pair${i}_a.wav" "$sd/00_prompt_original_spkA.wav"
  cp "$PREP_DIR/pair${i}_b.wav" "$sd/01_target_original_spkA.wav"
  gen="$gen_dir/pair${i}_concat_s2mel_style-none.wav"
  [ -f "$gen" ] && cp "$gen" "$sd/02_generated_step${STEP}.wav" || echo "WARN: step${STEP} missing pair${i}"
done
cat > "$PKG_DIR/README.txt" << README
exp05 valid set listening pack — same-speaker pairs — step${STEP}
Model: s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x (cont1 run 07law8n7)
Inference: style_mode=none, inference_steps=${INFER_STEPS}, cfg=${CFG}, temperature=0.7, seed=1234, prompt=first N seconds of A utterance

Files per sample:
  00_prompt_original_spkA.wav  — utterance A (reference/prompt, same speaker)
  01_target_original_spkA.wav  — utterance B (original audio, ground truth for synthesis)
  02_generated_step${STEP}.wav — exp05 step${STEP}: B content synthesized using A as prompt

Samples:
  sample0: laion_emolia ZH_B00031_S04396 (Chinese real speech)
  sample1: laion_emolia ZH_B00064_S08181 (Chinese real speech)
  sample2: StarRail 忘归人/Fugue (English game audio)
README
cd "$OUT_ROOT"
tar czf "exp05_valid_pair_listening_${DATE}_step${STEP}_steps${INFER_STEPS}_cfg0p7.tar.gz" "exp05_valid_pair_listening_${DATE}_step${STEP}_steps${INFER_STEPS}_cfg0p7/"
echo ""
echo "ALL_DONE: $OUT_ROOT/exp05_valid_pair_listening_${DATE}_step${STEP}_steps${INFER_STEPS}_cfg0p7.tar.gz"
sha256sum "exp05_valid_pair_listening_${DATE}_step${STEP}_steps${INFER_STEPS}_cfg0p7.tar.gz"
