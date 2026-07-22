# semantic2any

IndexTTS2.5-style semantic-to-mel training code. The model keeps the IndexTTS2
semantic2mel contract and replaces the CFM velocity estimator with a
ZipVoice-inspired ZipFormer stack.

## Install

Use `uv` from the repository root:

```bash
uv sync --frozen
```

The frozen feature-extractor and BigVGAN implementations used by this project
are included under `semantic2any/third_party/`; no IndexTTS, MaskGCT, SAC, or
BigVGAN source checkout is required. Model weights remain external and are
selected by workflow so a new machine does not need to download the full
IndexTTS bundle. See [`docs/model-assets.md`](docs/model-assets.md) for pinned,
minimal downloads and an offline deployment layout.

SAC support downloads only the pinned `zai-org/glm-4-voice-tokenizer` files
needed by the semantic encoder. It does not download SAC's acoustic modules or
complete codec checkpoint.

## Manifest

Training reads JSONL. Same-speaker pairing is enabled by default, so every row
must include `speaker_id`; `duration` is recommended so references shorter than
the configured three-second minimum can be filtered before loading:

```json
{"audio_path": "/path/to/audio.wav", "speaker_id": "speaker-001", "duration": 4.2}
```

Each target is paired at runtime with a different utterance carrying the same
`speaker_id`. The reference mel and style come from that prompt utterance, while
the generated suffix comes from the target. Their combined duration is capped
at 30 seconds: the prompt is shortened first but never below three seconds,
then the target is shortened if necessary. Set `data.pair_same_speaker: false`
to disable speaker pairing. With `data.random_split_audio: false`, this retains
the legacy single-utterance random-prefix behavior.

For local datasets organized as one audio subdirectory per source, the JSONL
loader also accepts the entire metadata directory:

```text
<dataset-root>/
  ears/*.flac
  expresso/*.flac
  metadata/ears.jsonl
  metadata/expresso.jsonl
```

Relative paths such as `"audio_path": "../ears/example.flac"` are resolved
relative to the JSONL file. Passing `metadata/` loads every `*.jsonl` in that
directory.

Optional precomputed fields are supported for faster iteration:

```json
{
  "mel_path": "/path/to/mel.pt",
  "semantic_path": "/path/to/semantic.pt",
  "style_path": "/path/to/style.pt",
  "prompt_len": 120
}
```

Tensor conventions:

- `mel`: `[n_mels, T]` (80 or 128 depending on the config)
- `semantic`: `[T_sem, 1024]` MaskGCT embeddings, `[T_sem, 1280]` SAC raw
  semantic embeddings, or `[Q, T_sem]` discrete codebooks
- `style`: `[192]`

## Semantic codec ablation

The default backend remains MaskGCT. Select SAC with one CLI option on training,
precomputation, inference, or paired rendering:

```bash
uv run accelerate launch trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --semantic-codec sac \
  --train-jsonl /path/to/train.jsonl \
  --output-dir exp/s2mel-sac
```

The two continuous feature contracts are:

- `maskgct`: approximately 50 Hz, 1024 dimensions
- `sac`: 12.5 Hz, 1280 dimensions, 16384-entry semantic codebook

`SAC-16k-62_5Hz` reports the combined codec rate: 12.5 Hz semantic plus 50 Hz
acoustic. This project intentionally uses only the raw semantic stream. It runs
the GLM-4-Voice quantizing encoder and codebook lookup used by SAC, and does not
download or instantiate SAC's acoustic encoder, decoder, or 2 GB codec
checkpoint.

The default tokenizer revision is pinned in `semantic_codec.revision`. Set
`semantic_codec.tokenizer_path` for a local snapshot, or configure
`semantic_codec.cache_dir` and `local_files_only` for the Hugging Face cache.
The resolved config records the codec, input dimension, frame rate, source
model, and fingerprint.

Precompute each ablation into its own directory:

```bash
uv run python scripts/precompute_s2mel_features.py \
  --config configs/s2mel_zipformer.yaml \
  --semantic-codec sac \
  --source /path/to/train.jsonl \
  --output-dir datasets/features-sac \
  --device cuda:0
```

`feature_metadata.json` and each manifest row carry the codec fingerprint.
Reusing a directory with another codec or tokenizer revision fails unless
`--overwrite` is given. MaskGCT and SAC checkpoints are not interchangeable
because their length-regulator projections have different input dimensions.

## SpeechData Shards

The ZipFormer trainer can also read normalized SpeechData outputs with this
layout:

```text
<dataset>/
  audio/<dataset>-000000.tar
  metadata/<dataset>-000000.jsonl
```

Each metadata row should contain `audio_path` like
`audio/<dataset>-000000.tar/<sample_id>.flac`, plus `speaker_id` and preferably
`duration` for same-speaker pairing. The trainer extracts tar members to a local
cache and then reuses the existing online IndexTTS feature extractor:

```bash
accelerate launch trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --train-speechdata-dir gs://noiz-taiwan-audio-data/preprocessed/expresso \
  --speechdata-cache-dir /tmp/semantic2any-speechdata \
  --output-dir exp/s2mel_zipformer-expresso
```

For local shards, pass the local dataset directory instead, for example
`/mnt/data_3t_2/datasets/raw_data/preprocessed/expresso`.

See `docs/gcs-datasets.md` for GCS authentication, dataset layout, and training
examples.

## Train

Run all training and other long-running jobs in a descriptively named `tmux`
session so they survive terminal or Cursor disconnections. Check existing
sessions first to avoid duplicate jobs:

```bash
tmux list-sessions
tmux new-session -s s2mel-train
```

Detach with `Ctrl-b d` and reconnect with `tmux attach -t s2mel-train`.

### Random prompt/target split dataset

The dataset under
`/mnt/data_3t_1/datasets/preprocess/s2mel-train-data` has a dedicated 44.1 kHz,
128-band, 512-hop config and launcher. It disables speaker pairing and randomly
splits each waveform before feature extraction. Both prompt and target contain
at least three seconds of source audio; mel and semantic features are extracted
independently for the two segments, and the style embedding is computed from
the prompt only. DataLoader workers decode and prefetch waveforms. Segments
sharing a source sample rate are padded into a batch, moved to the feature
adapter device, and passed through cached 44.1 kHz and 16 kHz GPU resamplers;
the padded outputs are trimmed back to their individual lengths.

The default launcher loads all `metadata/*.jsonl` files and uses four GPUs:

```bash
tmux new-session -s s2mel-random-split
bash scripts/train_s2mel_random_split.sh
```

Detach with `Ctrl-b d`. To train only EARS:

```bash
TRAIN_JSONL=/mnt/data_3t_1/datasets/preprocess/s2mel-train-data/metadata/ears.jsonl \
  bash scripts/train_s2mel_random_split.sh
```

The launcher supports `DATASET_ROOT`, `TRAIN_JSONL`, `CONFIG`,
`NUM_PROCESSES`, `NUM_MACHINES`, and `MAIN_PROCESS_PORT` environment
overrides. Its default config is
`configs/s2mel_zipformer_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml`.

```bash
accelerate launch trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --train-jsonl /path/to/train.jsonl \
  --output-dir exp/s2mel_zipformer
```

For a one-step smoke test:

```bash
accelerate launch trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --train-jsonl /path/to/tiny.jsonl \
  --batch-size 1 \
  --max-steps 1
```

Checkpoints contain an IndexTTS-style `net` dictionary:

```python
{
    "net": {
        "cfm": ...,
        "length_regulator": ...,
        "gpt_layer": ...  # only when enabled
    }
}
```

This is intended to make later inference integration close to the IndexTTS2
`s2mel.pth` loading path.

## Style ablation

See [`docs/style-ablation.md`](docs/style-ablation.md) for the portable
environment setup, matched style/no-style training commands, runtime style
masking, paired inference, resume procedure, and reporting checklist.

## Backbone ablation

See [`docs/backbone-ablation.md`](docs/backbone-ablation.md) for the
parameter-matched ZipFormer versus IndexTTS2-derived DiT experiment protocol,
portable launch commands, and checkpoint compatibility rules.
