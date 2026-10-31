# IndexTTS2.5 → Fish Drama3 discrete codes

Independent branch `index-fish-mtp`, based on `fish-s2tok` 53b5ff8.
This predicts ten **discrete codebooks**, not the summed continuous S used by
`FishS2Tok`. Existing trainers and running experiments are unchanged.

## Architecture and alignment

Reference: Fish Speech commit
[0f8a12e8 tower.py](https://github.com/fishaudio/fish-speech/blob/0f8a12e8a651062fdee9c33a894bdccd3a8513f1/fish_speech/models/text2semantic/fish_omni/tower.py).
This implementation borrows shared enrichment and sequential residual-codebook
prediction, not the released model weights or its complete architecture.

A causal temporal network consumes right-shifted previous Fish frames and attends
to Index memory and a separate reference audio-code sequence. A head predicts c0
(4096 classes), then shared enrichment and nine causal blocks predict c1..c9
(1024 classes each). Training teacher-forces lower codebooks and previous time
frames, computing equal-weight per-codebook CE over valid target frames only.
Inference uses its own previous predictions. No target Fish codes are required at
inference. The reference segment is assumed available in full before generation.

Two explicit configurations:

* `configs/index_fish_mtp.yaml`: **raw**, original 25Hz Index code embeddings,
  all history plus 80ms lookahead. No frozen Index decoder, hence no concealed
  noncausal decoder features. This is the input-streaming-compatible baseline.
* `configs/index_fish_mtp_offline.yaml`: **decoded**, unpadded per-utterance
  frozen FP32 decoder output at 50Hz plus a separate original-ID embedding
  attached to both decoded subframes. Full source attention; offline input only.

There is no source interpolation, pooling, length-normalization/stretching, or
forced one-Index-token-to-one-Fish-frame mapping. Fish clock = 44100/2048 Hz;
Index clock = 25Hz (decoded clock = 50Hz). Frame centers define positional
embeddings and a Gaussian time-distance attention bias (sigma 150ms, bounded
at -12); a learned offset is bounded to +/-100ms. These are nominal frame-bin
coordinates, not a claim of measured acoustic receptive-field centers.
Every valid source position is retained; the soft bias does not guarantee that
training preserves every piece of information. The raw stream mask additionally
requires a token's **end/availability time** <= target frame end + lookahead.
Both training and inference use exactly this mask. This is offline supervision
for a streaming mapping network, not a claim that the upstream Index tokenizer
itself encodes live audio causally.

Do not enable lookahead masking on full-utterance decoded features: the decoder
is noncausal and may already encode future context. The constructor rejects this
combination. Streaming decoded features requires a separate audited receptive-
field/context-buffer implementation.

## Data contract

Both codecs must encode the **same audio segment**, with the same starting time.
The user confirmed the existing Fish corpus uses the Drama3 codec weights and
codebook order. Fish codec checksum remains pinned in the config. Preserve Index
extraction provenance and use the matching decoder bundle/checksum in decoded mode.

JSONL example (binary ranges are little-endian uint16):

```json
{"duration_seconds": 1.0, "target": {"id":"dataset/segment-a","codes":"fish.bin","offset_bytes":0,"num_frames":22}, "reference":{"id":"dataset/segment-b","codes":"fish.bin","offset_bytes":440,"num_frames":30}, "index":{"id":"dataset/segment-a","codes":"index.bin","offset_bytes":0,"num_frames":25}}
```

Fish layout is `[T,10]`, Index layout `[N]`. Relative paths use `codes_root` or
manifest directory; absolute paths and GCS byte ranges (optional pinned
`generation`) work. `target.id == index.id` identifies an exact segment, not a
speaker. Reference must identify a different segment and code range. Optional
`start_seconds` must agree. `duration_seconds` comes from audio, not token count.
Lengths are independently checked against the fixed rates with endpoint-padding
tolerance (Index <=81ms, Fish <=2 frames). This catches gross mismatch, not an
incorrect upstream ID assignment: that must be audited during extraction/join.

No target or source is independently cropped. Oversize examples fail explicitly;
prepare matched whole segments upstream. Only reference prefix cropping is used.
This avoids changing Index decoder edge context by independently cropping codes.
The 60s setting is a safety limit, not a measured GPU batch-capacity result.

Join existing Fish pairs and an Index JSONL export by exact IDs:

```bash
python scripts/prepare_index_fish_manifest.py \
  --fish-pairs /data/fish-pairs.jsonl --index-records /data/index-records.jsonl \
  --work-dir /data/join-tmp --output /data/index-fish-train.jsonl
```

Each Index row needs `id,codes,num_frames,duration_seconds`, optional
`offset_bytes,generation,start_seconds,codes_root`. The join uses on-disk SQLite,
rejects duplicate source IDs/missing matches/duration mismatches, writes atomically,
and does not extract codecs or guess matches by filename. Join train and validation
separately, preserving existing speaker/segment isolation. Full corpus joining and
coverage verification have **not** been performed as part of this code change.

## Training and inference

Fill manifest/output paths. For decoded mode also pin `index.bundle` and its SHA256.
Raw mode needs no decoder weights, but its `index_vocab` must match stored IDs.
The initial 256-width, 2 condition + 2 shared + 9 residual blocks is an experimental
small baseline, not a tuned production model. AdamW, constant LR, BF16 (configurable),
Accelerate DDP/accumulation, finite loss/gradient checks, validation and checkpoints
are supported. All checkpoints are kept; plan storage before a long run.

```bash
accelerate launch --num_processes 8 trainers/train_index_fish_mtp.py \
  --config configs/index_fish_mtp.yaml
# resume requires identical configuration and world size
accelerate launch --num_processes 8 trainers/train_index_fish_mtp.py \
  --config configs/index_fish_mtp.yaml --resume /run/checkpoint-1000
python scripts/infer_index_fish_mtp.py \
  --checkpoint /run/checkpoint-1000/inference.pt --index-codes /data/index.npy \
  --reference-record /data/reference.json --output /tmp/fish-codes.npy \
  --duration-seconds 1.0 --device cuda
```

Add `--wav --fish-repo --fish-checkpoint` with their respective values to decode
using the existing frozen Fish S→post_module→upsample→waveform path. Reference
codes are prepended for decoder context and trimmed from the result. Waveform
export uses original duration to trim endpoint padding. Without explicit duration,
inference uses N/25; target length is ceil(duration*44100/2048), not a GT Fish
length. Thus an unknown final partial Index bin can add at most approximately one
code interval; exact original duration is preferable when available.

## Incremental codes API

```python
from semantic2any.models.index_fish_mtp import RawCodeStream
model.eval()  # raw mode, bounded lookahead
stream = RawCodeStream(model, reference_codes, reference_lengths)  # batch=1
new_fish_frames = stream.push(index_chunk.long())  # [N] -> [1,T,10]
tail = stream.push(last_chunk.long(), final=True, duration_seconds=audio_duration)
```

The stream retains source codes and generated history. It emits only frames whose
lookahead is ready; EOS flushes the tail without inventing future tokens. Greedy
streaming matches whole-sequence greedy generation under the same bounded mask.
Implementation recomputes prefixes; **no KV caching or real-time speed claim**.
The API streams discrete codes, not audio packets. Causal Fish waveform decoding
with chunk buffers/boundary tests is separate future work. 80ms is a configurable
lookahead budget, not end-to-end measured latency.

## Verification scope

`OMP_NUM_THREADS=1 python -m pytest -q tests/test_index_fish_mtp.py`

Tests cover future/time and codebook label leakage, padding and unpadded decoder
contracts, source access, fixed clock lengths, codebook CE gradients, synthetic
small-example overfit and free generation, prefix/full teacher-forcing agreement,
strict manifest checks, future Index invariance and chunk/whole stream agreement.
CPU integration additionally exercises real tiny Index decoder bundles, training,
validation, checkpoint/resume, and CLI generation. These establish plumbing and
trainability, **not voice quality or real-data convergence**. Before full training:
join/audit paired data, GPU BF16 capacity canary, short real-data overfit, then
same-reference listening/CER/speaker-similarity comparisons for raw vs decoded.

Reproducible integration check (CPU only, creates its own output directory):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/smoke_index_fish_mtp.py \
  --output-dir /tmp/index-fish-mtp-canary --ddp
```

2026-10-31 validation: 8 unit tests passed; raw + decoded training, 2-process CPU
DDP, validation, exact resume equivalence (maximum parameter difference 0), and
CLI uint16 `[5,10]` generation passed. Default raw baseline: 20,886,785 parameters.
No GPU capacity test, real-data quality run, or end-to-end waveform streaming test
was performed. See the smoke output's `result.json` for the integration receipt.
