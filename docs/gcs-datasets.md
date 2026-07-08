# GCS Dataset Access

This project can train directly from normalized SpeechData shards stored in GCS.

## Bucket

Normalized datasets live under:

```text
gs://noiz-taiwan-audio-data/preprocessed/
```

GCP access details:

- Project: `noiz-430406`
- Region: `ASIA-EAST1`
- Service account: `taiwan-audio-rw@noiz-430406.iam.gserviceaccount.com`
- Current local key file: `/mnt/data_sdd/hhy/SpeechData/gcs-key.json`
- Bucket access model: Uniform Bucket-Level Access

## Authentication

On this machine, authenticate with the existing SpeechData key:

```bash
gcloud auth activate-service-account \
  taiwan-audio-rw@noiz-430406.iam.gserviceaccount.com \
  --key-file=/mnt/data_sdd/hhy/SpeechData/gcs-key.json

gcloud config set project noiz-430406
```

For long-running training jobs, also export the environment variables used by
Python GCS clients:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/mnt/data_sdd/hhy/SpeechData/gcs-key.json"
export GOOGLE_CLOUD_PROJECT="noiz-430406"
```

If you need a repository-local key for another machine, copy it to
`./gcs-key.json` manually and keep it untracked. This repository ignores
`gcs-key.json` and `key.json` to avoid committing credentials.

Verify access with:

```bash
gcloud storage ls gs://noiz-taiwan-audio-data/preprocessed/
```

## Dataset Layout

Each dataset has its own child directory under `preprocessed/`:

```text
gs://noiz-taiwan-audio-data/preprocessed/
  <dataset>/
    audio/
      <dataset>-000000.tar
      <dataset>-000001.tar
      ...
    metadata/
      <dataset>-000000.jsonl
      <dataset>-000001.jsonl
      ...
```

Metadata shards are JSONL files. Each row should include an `audio_path` relative
to the dataset root, pointing into the matching tar shard:

```json
{
  "id": "expresso__sample_000001",
  "audio_path": "audio/expresso-000000.tar/expresso__sample_000001.flac",
  "text": "Example transcript.",
  "speaker_id": "expresso__speaker_001",
  "duration": 3.42,
  "sample_rate": 48000,
  "language": "en",
  "source": "manifest.jsonl#sample_000001"
}
```

The trainer downloads each GCS tar shard into `speechdata_cache_dir`, extracts
referenced tar members into the same cache, then passes local FLAC paths to the
existing online IndexTTS feature extractor.

## Training From GCS

Install project dependencies with `uv`:

```bash
uv pip install -e .
```

Run training with a dataset prefix:

```bash
accelerate launch trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --train-speechdata-dir gs://noiz-taiwan-audio-data/preprocessed/expresso \
  --speechdata-cache-dir /tmp/semantic2any-speechdata \
  --output-dir exp/s2mel_zipformer-expresso
```

Validation can use another SpeechData prefix:

```bash
accelerate launch trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --train-speechdata-dir gs://noiz-taiwan-audio-data/preprocessed/expresso \
  --valid-speechdata-dir gs://noiz-taiwan-audio-data/preprocessed/vctk \
  --speechdata-cache-dir /tmp/semantic2any-speechdata
```

For local normalized shards, pass the local dataset directory instead:

```bash
accelerate launch trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --train-speechdata-dir /mnt/data_3t_2/datasets/raw_data/preprocessed/expresso
```

## Useful GCS Commands

```bash
gcloud storage ls gs://noiz-taiwan-audio-data/preprocessed/
gcloud storage ls gs://noiz-taiwan-audio-data/preprocessed/<dataset>/metadata/
gcloud storage cp gs://noiz-taiwan-audio-data/preprocessed/<dataset>/metadata/<shard>.jsonl .
```

Do not rely on object ACLs for access control; the bucket uses IAM through
Uniform Bucket-Level Access.
