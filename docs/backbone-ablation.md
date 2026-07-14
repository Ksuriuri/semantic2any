# ZipFormer versus DiT backbone ablation

This experiment compares two independently trained, style-conditioned s2mel
estimators:

- `ZipFormer`: the existing ZipVoice-inspired estimator.
- `DiT`: an IndexTTS2-derived Transformer with a WaveNet output head.

The DiT support code is vendored under
`semantic2any/models/indextts_dit/` with IndexTTS2 attribution and its
Apache-2.0 license. A separate IndexTTS2 checkout is still required for frozen
feature extraction and BigVGAN, but not for the DiT estimator itself.

## Experimental controls

Use the same manifests, split, seed, optimizer, scheduler, learning rate,
weight decay, warmup, checkpoint selection rule, inference settings, and
number of optimizer updates for both runs. Both baseline configs enable the
192-dimensional CAMPPlus style condition.

The DiT configuration is intentionally aligned with the released IndexTTS2
S2M estimator:

```text
DiT: hidden_dim=512, num_heads=8, depth=13
WaveNet: hidden_dim=512, num_layers=8, kernel_size=5
80-mel estimator (configs/s2mel_dit.yaml): 98,187,344 parameters
128-mel estimator (44.1 kHz config): 98,310,272 parameters
80-mel ZipFormer, FFN=4096: 97,959,184 parameters (0.23% fewer)
128-mel ZipFormer, FFN=4096: 98,032,960 parameters (0.28% fewer)
```

`model_metadata.json` in every training output records the selected backbone
and estimator/CFM/model parameter counts. If DiT needs a smaller per-GPU batch
to fit memory, decrease `train.batch_size` and increase
`train.grad_accumulation` so the effective global batch remains equal:

```text
effective_global_batch = per_gpu_batch × grad_accumulation × num_processes
```

The supplied 44.1 kHz DiT config starts with batch 8 and four accumulation
steps; with four GPUs it matches the ZipFormer baseline's global batch of 128
(32 × 1 × 4). The 13-layer DiT uses substantially more memory than the former
5-layer ablation config, so reduce the per-GPU batch and increase accumulation
proportionally if this starting point does not fit.

The supplied ZipFormer configs are capacity-matched to the IndexTTS2-sized
DiT with `feedforward_dim=4096`. Do not force the ZipFormer layer count to 13,
because a local ZipFormer block and an IndexTTS2 Transformer block contain
different submodules. The matched configs instead keep `hidden_dim=512` and
the U-Net-like layer distribution `[2, 2, 4, 4, 4]`.

For an efficiency-oriented comparison, create a separately named ZipFormer
config with `feedforward_dim=1536`; it has 56,048,960 estimator parameters in
the 128-mel setup. Report that result separately from the capacity-matched
comparison.

## Interpreting the reported 4.8x S2M speedup

IndexTTS2.5 reports S2M RTF values of 0.081 for U-DiT and 0.017 for
Zipformer, a ratio of 4.76x. The paper does not publish the Zipformer layer
distribution, hidden dimension, FFN dimension, parameter count, or MAC count,
so an exact configuration cannot be recovered from the primary source.

For the simplified ZipFormer implemented in this repository, a static
full-rate-equivalent compute estimate gives:

```text
[2,2,4,4,4], FFN=4096: about 1.6x less compute than DiT-13 + WaveNet
[2,2,4,4,4], FFN=1536: about 2.4x to 2.6x less compute
[1,1,2,2,2], FFN=1536: about 4.65x to 5.0x less compute
[1,1,2,2,2], FFN=1024: about 5.1x to 5.7x less compute
```

The ranges cover approximately 1,000 to 2,500 mel frames. They are compute
proxies, not measured RTF, and do not account for kernel fusion, memory
traffic, framework overhead, or hardware utilization.

The most plausible local hypothesis for reproducing a roughly 4.8x backbone
speed difference is therefore `hidden_dim=512`,
`num_layers=[1,1,2,2,2]`, and `feedforward_dim=1536`, while retaining the
existing downsampling factors, heads, and convolution kernels. Secondary
summaries on the web claim an eight-block, 512-dimensional, FFN=1024 model for
IndexTTS2.5, but that detail is absent from the paper and released code, so it
must not be cited as confirmed architecture.

Keep this paper-speed hypothesis separate from the capacity-matched
FFN=4096 experiment. The two experiments answer different questions.

## Environment setup

On a new machine:

```bash
git clone <repository-url> semantic2any
cd semantic2any
uv sync --frozen
uv run python -m unittest discover -s tests -v
```

Set the local IndexTTS checkout and model paths in the selected config's
`paths` section, or pass `--indextts-root` and `--model-dir` when launching
training. For the 44.1 kHz config, also make `vocoder.cache_dir` valid on the
new machine.

## One-step smoke tests

Run both commands with a small valid manifest before a full job:

```bash
uv run accelerate launch --num_processes 1 trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --train-jsonl /data/manifests/tiny.jsonl \
  --output-dir exp/smoke-zipformer \
  --indextts-root /opt/index-tts \
  --model-dir /models/IndexTTS-2-vLLM \
  --batch-size 1 \
  --max-steps 1 \
  --no-wandb \
  --style-condition

uv run accelerate launch --num_processes 1 trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_dit.yaml \
  --train-jsonl /data/manifests/tiny.jsonl \
  --output-dir exp/smoke-dit \
  --indextts-root /opt/index-tts \
  --model-dir /models/IndexTTS-2-vLLM \
  --batch-size 1 \
  --max-steps 1 \
  --no-wandb \
  --style-condition
```

DiT initializes its non-causal attention cache from `DiT.block_size` before
training and inference. The provided `block_size: 8192` matches IndexTTS2. If
an input exceeds it, the run stops with a clear error; either shorten the
example or increase `DiT.block_size`.

## Full 44.1 kHz random-split training

Check existing sessions and GPUs before creating a long-running job:

```bash
tmux list-sessions
nvidia-smi
```

Start the ZipFormer reference:

```bash
tmux new-session -s s2mel-zipformer-reference

TRAIN_JSONL=/data/splits/seed1234/train.jsonl \
VALID_JSONL=/data/splits/seed1234/valid.jsonl \
CONFIG=configs/s2mel_zipformer_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml \
OUTPUT_DIR=exp/s2mel-zipformer-reference \
NUM_PROCESSES=4 \
bash scripts/train_s2mel_random_split.sh \
  --indextts-root /opt/index-tts \
  --model-dir /models/IndexTTS-2-vLLM \
  --style-condition
```

Start the DiT comparison on the same resources, or after the reference run
finishes:

```bash
tmux new-session -s s2mel-dit-reference

TRAIN_JSONL=/data/splits/seed1234/train.jsonl \
VALID_JSONL=/data/splits/seed1234/valid.jsonl \
CONFIG=configs/s2mel_dit_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml \
OUTPUT_DIR=exp/s2mel-dit-reference \
NUM_PROCESSES=4 \
bash scripts/train_s2mel_random_split.sh \
  --indextts-root /opt/index-tts \
  --model-dir /models/IndexTTS-2-vLLM \
  --style-condition
```

Detach with `Ctrl-b d`; reconnect with:

```bash
tmux attach -t s2mel-zipformer-reference
tmux attach -t s2mel-dit-reference
```

## Matched inference

Each checkpoint must be constructed with its own resolved config:

```bash
uv run python scripts/infer_s2mel_zipformer.py \
  --config exp/s2mel-zipformer-reference/config.resolved.yaml \
  --checkpoint exp/s2mel-zipformer-reference/s2mel_final.pth \
  --input /data/eval/audio \
  --output-dir outputs/zipformer-reference \
  --style-mode reference \
  --inference-cfg-rate 0 \
  --seed 1234

uv run python scripts/infer_s2mel_zipformer.py \
  --config exp/s2mel-dit-reference/config.resolved.yaml \
  --checkpoint exp/s2mel-dit-reference/s2mel_final.pth \
  --input /data/eval/audio \
  --output-dir outputs/dit-reference \
  --style-mode reference \
  --inference-cfg-rate 0 \
  --seed 1234
```

Start with `inference_cfg_rate=0` to isolate the backbone comparison. If the
production protocol uses nonzero CFG, repeat both commands with the same CFG
value and report it separately. The paired mel comparison script supports the
same resolved configs and `--style-mode`.

## Checkpoint and style rules

- ZipFormer and DiT checkpoints are incompatible. Do not cross-load, cross
  resume, or change only `dit_type` in a config to reuse a checkpoint.
- Training rejects a weights-only or Accelerator resume when its saved resolved
  config records a different backbone.
- `--style-condition` and `--no-style-condition` route to the selected
  backbone. For DiT, they update both `DiT.style_condition` and
  `wavenet.style_condition`.
- The primary backbone comparison uses `--style-condition` for both models.
  The style/no-style training ablation and same-checkpoint `--style-mode none`
  diagnostic remain documented in [style-ablation.md](style-ablation.md).

## Report with each result

Record the git revision, resolved config, `model_metadata.json`, training and
evaluation manifests, seed, effective global batch, checkpoint step, estimator
parameter count, training throughput/GPU hours, inference steps, temperature,
CFG rate, and all objective/listening-test metrics. Report the capacity-matched
backbone result separately from the style ablation result.
