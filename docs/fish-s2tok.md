# Fish S2Tok: semantic code feature to summed tokenizer features

## Contract

Based on `feat/indextts25-codec-branch` at `fb0253d`. This is a separate training/inference path; existing S2VAE behavior is unchanged.

- Conditions: codebook 0 (4096 entries) lookup followed by its frozen `out_proj`: `[B,T,1024]`.
- Target: frozen projected semantic feature plus the sum of nine frozen projected residual codebook features, `S`, `[B,T,1024]`.
- Reference: another reference utterance's full summed `S`, `[B,Tr,1024]`.
- All features are at 44100/2048 ≈ 21.533 frames/s. There is no interpolation, pack2, or temporal downsampling.
- S is **before** the tokenizer's post-module and 4x upsampling. Waveform inference applies `post_module -> upsample -> decoder` with frozen weights.
- Fish checkpoint SHA256 is checked before use. Exact version matches `fish_drama3_codes_v1`. The current lookup computes FP32 tables from each quantizer's `out_proj(decode_code(ids))`; codec parameters are not optimized.

The baseline reuses the existing DiT/CFM with hidden=512, depth=13, heads=8. Target/noise/reference channels and the output head are 1024, while the semantic condition is linearly projected to 512. Reference S is a prompt prefix; its semantic condition is zero and its loss is masked. Padding is masked. The model predicts full S, not residual S minus codebook 0. CFM loss averages over valid target frames and channels. `model.feat_scale` is a fixed, invertible global scale (default 1); no per-utterance normalization is applied. Old 128-channel S2VAE checkpoints are not compatible with this model.

## Data

GCS source format: `preprocessed/<dataset>/features/fish_drama3_codes_v1/<shard>.u2.bin` and matching JSONL. Codes are little-endian uint16, `[T,10]`. This implementation reads byte ranges directly from local files or GCS (application-default credentials).

Training consumes an **explicit paired JSONL**. The dataset does not guess reference speakers from filenames or silently use the target as its own reference. Prepare same-speaker pairs and independent speaker-held-out validation manifests upstream; exclude any ineligible reference audio during that preparation. No full-corpus pairing or data download has been performed here.

```json
{"target":{"id":"target-id","codes":"gs://bucket/path/shard.u2.bin","offset_bytes":0,"num_frames":200,"num_codebooks":10,"dtype":"uint16","layout":"T,K"},"reference":{"id":"reference-id","codes":"gs://bucket/path/other.u2.bin","offset_bytes":4000,"num_frames":100,"num_codebooks":10,"dtype":"uint16","layout":"T,K"}}
```

`fish_code_path` is also accepted. Relative paths are resolved against the paired manifest's directory, or the row's `codes_root`. Different IDs alone do not prove different/same-speaker audio: pairing manifests remain responsible for identity and leakage checks. Oversize samples are deterministically prefix-cropped to configured maximum frames; semantic and target S use the exact same crop. Defaults: target 512 frames (~23.8s), reference 172 (~8s). Byte offsets are not waveform sample offsets. GCS requests per sample are supported for correctness; prefetch/local caching and a large-corpus sharded loader should be benchmarked before a 326M-row production run.

## Training

Use Python >=3.11, the repository's compatible PyTorch/Accelerate/OmegaConf environment, and the pinned Fish source/dependencies (hydra-core, descript-audio-codec, etc.). Optional GCS loading needs google-cloud-storage. Audio export needs soundfile. Official source used during validation: Fish commit `541566932abbaebcb0a40e562a1f846738383f38`.

Copy `configs/fish_s2tok_dit.yaml` and set codec repo/checkpoint, paired train/validation manifests, and a new output directory. Batch/LR are initial configurable values, not a validated production recipe.

```bash
accelerate launch --num_processes 8 --multi_gpu trainers/train_fish_s2tok.py --config /path/to/run.yaml
# Full model/optimizer/scaler/RNG state and batch position:
accelerate launch --num_processes 8 --multi_gpu trainers/train_fish_s2tok.py --config /path/to/run.yaml --resume exp/fish_s2tok_dit/checkpoint-1000
```

The loader keeps fixed manifest order; shuffle paired manifests reproducibly upstream. Resume requires the same resolved config and unchanged manifests/code files; distributed topology must also remain the same. DataLoader offsets allow batch skipping. Checkpoints are retained without automatic deletion and contain Accelerate state, `trainer.json`, and a standalone `inference.pt`. Keep sufficient disk space. Validation uses fixed noise seeds and only the configured number of validation batches; this is sampled flow loss, not a perceptual-quality evaluation. No auxiliary waveform/perceptual loss is enabled.

## Inference

```bash
python scripts/infer_fish_s2tok.py \
  --checkpoint exp/fish_s2tok_dit/checkpoint-1000/inference.pt \
  --semantic-codes semantic0.npy --reference-record reference.json \
  --fish-repo /path/to/fish-speech --fish-checkpoint /path/to/codec.pth \
  --output predicted_S.npy --wav predicted.wav --steps 25 --cfg 0.7
```

`semantic0.npy` must be integer `[T]`, IDs 0..4095. `reference.json` is a single full-10-codebook record as above. Output S is float `[T,1024]`. Inference currently supports one variable-length utterance at a time and rejects sequences beyond the configured DiT context. Reference S is prepended for causal Fish decoding, then its waveform prefix is removed. Output waveform length is code-frame-quantized (`T*2048` samples); original unpadded duration is not inferable from codebook 0 alone. Generated values are not re-quantized. FLOAT WAV export avoids hidden clipping or loudness normalization.

## Validation scope

Unit tests cover code ranges, malformed ranges, frozen codebook summation, different reference/target lengths, masked padding, finite gradients, and inference shape. Real-codec tests compare reconstructed S and waveform against official `from_codes`/`from_indices`. Tiny CPU training/resume/inference smoke tests check execution only; they do not demonstrate trained audio quality, production throughput, or full-size multi-GPU convergence. No new production training is started by this change.

Validated on E0904 without CUDA allocation: 3 unit tests passed; real EARS codes `[1,12,10]` gave S max error `9.54e-7` and waveform max error `1.15e-7` against official codec operations. A tiny CPU model trained/saved at step 1, restored optimizer/RNG state and reached step 2, exported finite S `[12,1024]` and a finite 24,576-sample 44.1 kHz waveform. A two-rank CPU Accelerate run also completed training/validation/checkpointing (world_size=2). These use tiny explicit paired fixtures, not a held-out quality evaluation. Smoke data/artifacts are not committed.
