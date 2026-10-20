# October 2026 S2VAE experiment snapshots

These configurations preserve actual E0904 experiment settings; they are not portable defaults or claims of model quality. Core trainer/model code is unchanged by this archival commit.

## DiT

- `s2vae_dit_feature_resume440000_20261019_3ep.yaml`: current run, resumed the complete model/optimizer state from 440000 with decoder feature loss weight 0.5. Batch 12 × 8 GPUs × accumulation 6 = 576, LR 5e-5; 128-frame feature crops with 16-frame context. Global epochs=12 means three cumulative fine-tuning epochs, not twelve additional epochs. Validation at 442000: 0.69789; one point does not establish a trend.
- `s2vae_dit_finetune_20261016_feature_from432548_b12a6_2ep.yaml`: preceding feature-loss fine-tune.
- `s2vae_dit_finetune_flowonly_resume440500_3ep.yaml`: **FAILED / STOPPED experiment**, retained for comparison only. Removing feature loss was followed by gradient spikes and worsening validation; stopped at 450400. Do not use as a recommended training recipe. The original 440500 checkpoint was removed by checkpoint retention and is no longer available.

The current run uses a separate output directory. Protected `_resume/checkpoint-440000` and the new run's `_resume/checkpoint-440500` were verified against their sources; the latter is not the original deleted checkpoint. Checkpoints themselves are not in Git.

## ZipFormer

Archived configs include the base model, October 9/10/14/15 flow/feature fine-tunes, and feature-run resumes at 522000 and 526000. They preserve their historical resume/output paths; copy and edit the config before starting a different run, so outputs cannot overwrite an earlier experiment.

## Paths and tools

Configs retain original `/mnt/data_3t_2/...` manifest paths and some `/mnt/data_sdd/...` checkpoint paths. These mounts, source manifests, tokenizer/AudioVAE weights, and checkpoint state must be provisioned separately on another machine. Replace data, weight, resume and output paths before use; do not modify a live experiment config to launch a new run.

- `scripts/train_s2vae_zipformer.sh`: accepts CONFIG, TRAIN_JSONL and OUTPUT_DIR environment variables and forwards extra Accelerate/trainer arguments as implemented.
- `t2s_align/build_s2vae_sr_balanced_split.py`: deterministic sample-rate balancing, configurable source/output and duration targets.
- `t2s_align/build_s2vae_equalized_5k_split.py`: five-rate split policy preserving pairable speaker groups. Its `LOOKUP_PATH` constant points to the original E0904 IndexTTS decoder artifact; adjust it for another installation while retaining the required artifact hash.
- `t2s_align/validate_s2vae_equalized_split.py`: split validation and sampled binary checks.
- `t2s_align/infer_indextts25_s2mel_compare.py`: comparison inference; override `--repo` for the local IndexTTS checkout. Requires the compatible inference environment and model assets.

Machine-specific, checkpoint-by-checkpoint HTML builders, generated media, logs, backup files, and one-off trial configurations are deliberately excluded. The archived launch/config files do not start a process merely by being committed.
