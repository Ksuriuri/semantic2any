# Style feature ablation

This guide reproduces the style ablation on a fresh machine without maintaining
separate training or inference scripts. It covers two different questions:

1. **Training ablation:** does a model trained without the style channel perform
   as well as the normal model?
2. **Inference diagnostic:** how much does an existing style-conditioned
   checkpoint depend on its reference style embedding?

These results are not interchangeable. The training ablation requires a new
checkpoint, while the inference diagnostic reuses the same checkpoint and masks
the projected style embedding at runtime.

## 1. Set up a new machine

Clone the same revision, create the locked environment, and verify the unit
tests:

```bash
git clone <repository-url> semantic2any
cd semantic2any
uv sync --frozen
uv run pytest -q
```

The feature implementation is included in this repository. Download the
minimal model assets described in `docs/model-assets.md`, then update
`paths.model_dir` or pass `--model-dir`. BigVGAN is needed only for inference.

Use absolute paths for manifests when moving between machines. A training row
must at least contain:

```json
{"audio_path": "/data/audio/example.wav", "speaker_id": "speaker-001", "duration": 6.2}
```

For the random-split setup, `speaker_id` is not required because
`pair_same_speaker` is disabled.

## 2. Run a one-step smoke test

Run both variants before starting a long job. Use a small but valid training
manifest whose audio satisfies the prompt/target duration constraints.

Style-conditioned:

```bash
uv run accelerate launch --num_processes 1 trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --train-jsonl /data/manifests/tiny.jsonl \
  --output-dir exp/smoke-style-reference \
  --model-dir checkpoints/feature-extractors \
  --batch-size 1 \
  --max-steps 1 \
  --no-wandb \
  --style-condition
```

No-style:

```bash
uv run accelerate launch --num_processes 1 trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --train-jsonl /data/manifests/tiny.jsonl \
  --output-dir exp/smoke-style-none \
  --model-dir checkpoints/feature-extractors \
  --batch-size 1 \
  --max-steps 1 \
  --no-wandb \
  --no-style-condition
```

Check each output directory for `config.resolved.yaml` and confirm:

```yaml
s2mel:
  ZipFormer:
    style_condition: true   # reference run
```

or:

```yaml
s2mel:
  ZipFormer:
    style_condition: false  # no-style run
```

## 3. Launch the two full training runs

Keep the data split, seed, batch size, number of updates, optimizer, and all
non-style settings identical. Use different output directories and W&B run
names. The CLI override is saved into `config.resolved.yaml` and every
compatible checkpoint.

Long-running jobs must run in named `tmux` sessions. Check for existing jobs
first:

```bash
tmux list-sessions
nvidia-smi
```

For the random-split 44.1 kHz setup, start the reference run:

```bash
tmux new-session -s s2mel-style-reference

TRAIN_JSONL=/data/splits/seed1234/train.jsonl \
VALID_JSONL=/data/splits/seed1234/valid.jsonl \
CONFIG=configs/s2mel_zipformer_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml \
OUTPUT_DIR=exp/s2mel-style-reference \
NUM_PROCESSES=4 \
bash scripts/train_s2mel_random_split.sh \
  --model-dir checkpoints/feature-extractors \
  --style-condition
```

Detach with `Ctrl-b d`. Start the no-style run on the intended GPUs or after the
reference run finishes:

```bash
tmux new-session -s s2mel-style-none

TRAIN_JSONL=/data/splits/seed1234/train.jsonl \
VALID_JSONL=/data/splits/seed1234/valid.jsonl \
CONFIG=configs/s2mel_zipformer_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml \
OUTPUT_DIR=exp/s2mel-style-none \
NUM_PROCESSES=4 \
bash scripts/train_s2mel_random_split.sh \
  --model-dir checkpoints/feature-extractors \
  --no-style-condition
```

Reconnect with:

```bash
tmux attach -t s2mel-style-reference
tmux attach -t s2mel-style-none
```

Do not initialize the no-style run from a style-conditioned checkpoint, or the
reverse. `style_condition` changes the decoder input width, so those checkpoint
architectures are not compatible.

For an exact interrupted-run resume, use the matching Accelerator checkpoint
directory and the run's resolved config:

```bash
uv run accelerate launch trainers/train_s2mel_zipformer.py \
  --config exp/s2mel-style-none/config.resolved.yaml \
  --train-jsonl /data/splits/seed1234/train.jsonl \
  --valid-jsonl /data/splits/seed1234/valid.jsonl \
  --output-dir exp/s2mel-style-none \
  --resume-from exp/s2mel-style-none/checkpoint-10000 \
  --no-style-condition
```

## 4. Run matched inference

Always construct a model with the `config.resolved.yaml` saved by that run.
Changing only `style_condition` in a YAML file cannot convert an existing
checkpoint.

Evaluate the normal model with reference style:

```bash
uv run python scripts/infer_s2mel_zipformer.py \
  --config exp/s2mel-style-reference/config.resolved.yaml \
  --checkpoint exp/s2mel-style-reference/s2mel_final.pth \
  --input /data/eval/audio \
  --output-dir outputs/style-reference \
  --style-mode reference \
  --seed 1234 \
  --inference-cfg-rate 0
```

Evaluate the independently trained no-style model:

```bash
uv run python scripts/infer_s2mel_zipformer.py \
  --config exp/s2mel-style-none/config.resolved.yaml \
  --checkpoint exp/s2mel-style-none/s2mel_final.pth \
  --input /data/eval/audio \
  --output-dir outputs/style-none-trained \
  --style-mode reference \
  --seed 1234 \
  --inference-cfg-rate 0
```

`style-mode` has no effect when the checkpoint architecture has
`style_condition: false`; the model has no style input channel.

## 5. Diagnose one checkpoint with and without style

Use the same style-conditioned checkpoint, input, seed, temperature, number of
steps, and CFG rate for both commands. `--style-mode none` masks style **after**
`style_projection`, including its bias.

```bash
uv run python scripts/infer_s2mel_zipformer.py \
  --config exp/s2mel-style-reference/config.resolved.yaml \
  --checkpoint exp/s2mel-style-reference/s2mel_final.pth \
  --input /data/eval/audio \
  --output-dir outputs/same-checkpoint-ablation \
  --style-mode reference \
  --seed 1234 \
  --inference-cfg-rate 0

uv run python scripts/infer_s2mel_zipformer.py \
  --config exp/s2mel-style-reference/config.resolved.yaml \
  --checkpoint exp/s2mel-style-reference/s2mel_final.pth \
  --input /data/eval/audio \
  --output-dir outputs/same-checkpoint-ablation \
  --style-mode none \
  --seed 1234 \
  --inference-cfg-rate 0
```

The generated names include `style-reference` or `style-none`, so the second
command does not overwrite the first. Start with CFG 0 to isolate style. If the
deployed system uses nonzero CFG, repeat both commands with the same production
CFG value and report that result separately.

## 6. Paired mel comparison

Run both modes against the same pair manifest:

```bash
uv run python scripts/render_paired_mel_comparisons.py \
  --config exp/s2mel-style-reference/config.resolved.yaml \
  --checkpoint exp/s2mel-style-reference/s2mel_final.pth \
  --pair-manifest /data/eval/pairs.jsonl \
  --output-dir outputs/paired-mel-reference \
  --style-mode reference \
  --inference-cfg-rate 0 \
  --seed 1234

uv run python scripts/render_paired_mel_comparisons.py \
  --config exp/s2mel-style-reference/config.resolved.yaml \
  --checkpoint exp/s2mel-style-reference/s2mel_final.pth \
  --pair-manifest /data/eval/pairs.jsonl \
  --output-dir outputs/paired-mel-none \
  --style-mode none \
  --inference-cfg-rate 0 \
  --seed 1234
```

Image names and `mel_visualization_manifest.jsonl` record the selected style
mode. Keep separate output directories because the manifest file is rewritten
on each invocation.

## 7. Reporting checklist

Record the following with every result:

- Git revision and resolved config.
- Checkpoint step and selection criterion.
- Train/validation/evaluation manifests and seed.
- `style_condition` for training.
- `style_mode`, inference steps, temperature, and CFG rate for inference.
- Whether the result is a separately trained architecture ablation or a
  same-checkpoint runtime diagnostic.

The primary evidence for whether style helps is the matched comparison between
the independently trained reference and no-style models. The same-checkpoint
masking result only shows how the reference model reacts when a conditioning
signal it saw during training is removed.
