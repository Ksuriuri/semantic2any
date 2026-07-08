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

Training reads JSONL. The simplest record is:

```json
{"audio_path": "/path/to/audio.wav"}
```

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

- `mel`: `[80, T]`
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
`audio/<dataset>-000000.tar/<sample_id>.flac`. The trainer extracts tar members
to a local cache and then reuses the existing online IndexTTS feature extractor:

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
