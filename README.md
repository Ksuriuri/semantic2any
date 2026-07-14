# semantic2any

IndexTTS2.5-style semantic-to-mel training code. The model keeps the IndexTTS2
semantic2mel contract and replaces the CFM velocity estimator with a
ZipVoice-inspired ZipFormer stack.

## Install

Use `uv` from the repository root:

```bash
uv pip install -e .
```

The training code imports frozen feature extractors from a local IndexTTS
checkout. By default it expects `/mnt/data_sdd/hhy/index-tts`; override this with
`--indextts-root` or `paths.indextts_root` in `configs/s2mel_zipformer.yaml`.

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
- `semantic`: `[T_sem, 1024]` continuous semantic embeddings or `[Q, T_sem]`
  discrete codebooks when using a discrete length regulator
- `style`: `[192]`

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
