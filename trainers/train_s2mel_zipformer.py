from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from semantic2any.data.s2mel_dataset import (
    S2MelCollator,
    S2MelInMemoryDataset,
    S2MelJsonlDataset,
    S2MelSpeakerPairedDataset,
    S2MelSpeechDataDataset,
)
from semantic2any.models import Semantic2MelModel
from semantic2any.utils.checkpoint import load_compatible_checkpoint, save_compatible_checkpoint
from semantic2any.utils.indextts_adapters import IndexTTSFeatureAdapter, move_feature_batch_to_device


def _get(obj, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def _optional_float(value) -> float | None:
    return None if value in (None, "None") else float(value)


def _dit_type(cfg) -> str:
    return str(_get(cfg.s2mel, "dit_type", "ZipFormer"))


def _set_style_condition(cfg, enabled: bool) -> None:
    dit_type = _dit_type(cfg)
    if dit_type == "ZipFormer":
        cfg.s2mel.ZipFormer.style_condition = enabled
        return
    if dit_type == "DiT":
        cfg.s2mel.DiT.style_condition = enabled
        cfg.s2mel.wavenet.style_condition = enabled
        return
    raise ValueError(f"Unsupported s2mel.dit_type={dit_type!r} for style override")


def model_parameter_metadata(model, cfg) -> dict[str, int | str]:
    cfm = model.models["cfm"]
    return {
        "dit_type": _dit_type(cfg),
        "estimator_parameters": sum(parameter.numel() for parameter in cfm.estimator.parameters()),
        "cfm_parameters": sum(parameter.numel() for parameter in cfm.parameters()),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def validate_resume_backbone(cfg, resume_path: Path | None) -> None:
    """Fail early instead of silently partially loading another backbone."""
    if resume_path is None:
        return
    checkpoint_config = None
    if resume_path.is_file():
        checkpoint_config = torch.load(resume_path, map_location="cpu").get("config")
    elif resume_path.is_dir():
        resolved_config = resume_path.parent / "config.resolved.yaml"
        if resolved_config.is_file():
            checkpoint_config = OmegaConf.load(resolved_config)
    if checkpoint_config is None:
        return
    checkpoint_s2mel = _get(checkpoint_config, "s2mel")
    checkpoint_dit_type = str(_get(checkpoint_s2mel, "dit_type", "ZipFormer"))
    if checkpoint_dit_type != _dit_type(cfg):
        raise ValueError(
            f"Cannot resume {checkpoint_dit_type} checkpoint with {_dit_type(cfg)} config. "
            "Backbone checkpoints are not compatible."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IndexTTS2.5-style semantic2mel estimator.")
    parser.add_argument("--config", default="configs/s2mel_zipformer.yaml")
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--valid-jsonl", default=None)
    parser.add_argument("--train-speechdata-dir", default=None)
    parser.add_argument("--valid-speechdata-dir", default=None)
    parser.add_argument("--speechdata-cache-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--indextts-root", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume-epoch-step", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    style_group = parser.add_mutually_exclusive_group()
    style_group.add_argument(
        "--style-condition",
        dest="style_condition",
        action="store_true",
        default=None,
        help="Include the CAMPPlus style channel in the selected estimator.",
    )
    style_group.add_argument(
        "--no-style-condition",
        dest="style_condition",
        action="store_false",
        default=None,
        help="Train a no-style baseline from scratch.",
    )
    return parser.parse_args()


def apply_overrides(cfg, args: argparse.Namespace):
    if args.train_jsonl is not None:
        cfg.data.train_jsonl = args.train_jsonl
    if args.valid_jsonl is not None:
        cfg.data.valid_jsonl = args.valid_jsonl
    if args.train_speechdata_dir is not None:
        cfg.data.train_speechdata_dir = args.train_speechdata_dir
    if args.valid_speechdata_dir is not None:
        cfg.data.valid_speechdata_dir = args.valid_speechdata_dir
    if args.speechdata_cache_dir is not None:
        cfg.data.speechdata_cache_dir = args.speechdata_cache_dir
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    if args.indextts_root is not None:
        cfg.paths.indextts_root = args.indextts_root
    if args.model_dir is not None:
        cfg.paths.model_dir = args.model_dir
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.resume_from is not None:
        cfg.train.resume_from = args.resume_from
    if args.no_wandb:
        cfg.train.no_wandb = True
    if args.style_condition is not None:
        _set_style_condition(cfg, args.style_condition)
    return cfg


def cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """Cosine LR with warmup, clamped at the configured minimum.

    Unlike transformers.get_cosine_schedule_with_warmup, stepping past
    num_training_steps (e.g. after a resume replay) keeps the LR at its
    minimum instead of climbing back up the cosine curve.
    """
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}")

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def constant_schedule_with_warmup(optimizer, num_warmup_steps: int):
    """Linear warmup followed by a constant learning rate."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_lr_scheduler(optimizer, cfg, num_training_steps: int):
    schedule = str(_get(cfg.train, "lr_scheduler", "cosine")).lower()
    warmup_steps = int(cfg.train.warmup_steps)
    if schedule == "cosine":
        learning_rate = float(cfg.train.learning_rate)
        min_learning_rate = float(_get(cfg.train, "min_learning_rate", 1.0e-5))
        if learning_rate <= 0.0:
            raise ValueError(f"train.learning_rate must be positive, got {learning_rate}")
        if not 0.0 <= min_learning_rate <= learning_rate:
            raise ValueError(
                "train.min_learning_rate must be between 0 and train.learning_rate; "
                f"got {min_learning_rate} and {learning_rate}"
            )
        return cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=min_learning_rate / learning_rate,
        )
    if schedule == "constant_with_warmup":
        return constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    raise ValueError(
        f"Unsupported train.lr_scheduler={schedule!r}; "
        "expected 'cosine' or 'constant_with_warmup'"
    )


def make_source_dataset(cfg, source: str, *, speechdata: bool = False) -> Dataset:
    if speechdata:
        return S2MelSpeechDataDataset(
            source,
            cache_dir=_get(cfg.data, "speechdata_cache_dir", None),
        )
    return S2MelJsonlDataset(source)


def make_dataloader(
    cfg,
    source: str,
    shuffle: bool,
    *,
    speechdata: bool = False,
    persistent_workers: bool = True,
    dataset: Dataset | None = None,
) -> DataLoader:
    if dataset is None:
        dataset = make_source_dataset(cfg, source, speechdata=speechdata)
    spect = cfg.preprocess_params.spect_params
    collator = S2MelCollator(
        hop_length=int(spect.hop_length),
        sample_rate=int(cfg.preprocess_params.sr),
        min_prompt_seconds=float(cfg.data.min_prompt_seconds),
        max_prompt_seconds=_optional_float(_get(cfg.data, "max_prompt_seconds", 5.0)),
        min_generated_frames=int(cfg.data.min_generated_frames),
        min_target_seconds=_optional_float(_get(cfg.data, "min_target_seconds", None)),
        max_pair_seconds=float(_get(cfg.data, "max_pair_seconds", 30.0)),
        min_pair_prompt_seconds=float(_get(cfg.data, "min_pair_prompt_seconds", 3.0)),
        decode_audio_in_worker=bool(_get(cfg.data, "decode_audio_in_worker", False)),
        skip_audio_errors=bool(_get(cfg.data, "skip_audio_errors", False)),
        max_audio_seconds=_optional_float(_get(cfg.data, "max_audio_seconds", None)),
    )
    kwargs: dict[str, Any] = {}
    if int(cfg.data.num_workers) > 0:
        kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
        # Validation loaders re-create workers per pass so that the seeded RNG
        # fork in validate() also controls worker seeding (deterministic prompts).
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=shuffle,
        num_workers=int(cfg.data.num_workers),
        collate_fn=collator,
        pin_memory=True,
        drop_last=shuffle,
        **kwargs,
    )


def _split_source(cfg, split: str) -> tuple[str, bool]:
    speechdata_source = str(_get(cfg.data, f"{split}_speechdata_dir", "") or "")
    if speechdata_source:
        return speechdata_source, True
    return str(_get(cfg.data, f"{split}_jsonl", "") or ""), False


@torch.no_grad()
def preload_dataset_features(
    dataset: Dataset,
    *,
    split: str,
    cfg,
    adapter: IndexTTSFeatureAdapter,
    accelerator: Accelerator,
) -> S2MelInMemoryDataset:
    """Extract each utterance once and retain compact features in CPU RAM.

    Every DDP rank keeps its own copy so shuffled sampling never causes cache
    misses or cross-process synchronization during training.
    """

    batch_size = max(
        1,
        int(_get(cfg.data, "preload_batch_size", _get(cfg.data, "feature_batch_size", 16))),
    )
    records: list[dict[str, Any]] = []
    feature_bytes = 0
    total = len(dataset)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        source_records = [dataset[index] for index in range(start, end)]
        audio_paths = [record.get("audio_path") for record in source_records]
        if any(not isinstance(path, str) or not path for path in audio_paths):
            raise ValueError(f"{split} preload encountered a record without audio_path")
        features = adapter.extract_utterance_features(audio_paths)
        for source_record, feature in zip(source_records, features, strict=True):
            mel = feature["mel"].detach().to(device="cpu", dtype=torch.float16).contiguous()
            semantic = feature["semantic"].detach().to(device="cpu", dtype=torch.float16).contiguous()
            style = feature["style"].detach().to(device="cpu", dtype=torch.float32).contiguous()
            record = dict(source_record)
            record.update({"mel": mel, "semantic": semantic, "style": style})
            records.append(record)
            feature_bytes += sum(tensor.numel() * tensor.element_size() for tensor in (mel, semantic, style))
        if accelerator.is_main_process and (end == total or end % (batch_size * 10) == 0):
            print(f"[Preload] {split}: {end}/{total} utterances")

    if accelerator.is_main_process:
        print(
            f"[Preload] {split}: loaded {len(records)} utterances into "
            f"{feature_bytes / (1024**3):.2f} GiB CPU RAM per rank"
        )
    return S2MelInMemoryDataset(records)


def make_speaker_paired_dataset(
    cfg,
    dataset: Dataset,
    *,
    fallback_prompt_dataset: Dataset | None = None,
) -> S2MelSpeakerPairedDataset:
    spect = cfg.preprocess_params.spect_params
    return S2MelSpeakerPairedDataset(
        dataset,
        min_prompt_seconds=float(_get(cfg.data, "min_pair_prompt_seconds", 3.0)),
        hop_length=int(spect.hop_length),
        sample_rate=int(cfg.preprocess_params.sr),
        fallback_prompt_dataset=fallback_prompt_dataset,
    )


def _weight_checkpoint_step(path: Path) -> int | None:
    name = path.name
    if not name.startswith("s2mel_step") or not name.endswith(".pth"):
        return None
    try:
        return int(name.removeprefix("s2mel_step").removesuffix(".pth"))
    except ValueError:
        return None


def rotate_checkpoints(
    output_dir: Path,
    keep_last: int,
    *,
    archive_interval: int = 0,
) -> None:
    if keep_last <= 0:
        return

    def is_archived(step: int) -> bool:
        return archive_interval > 0 and step % archive_interval == 0

    regular_checkpoints = sorted(
        (
            (step, path)
            for path in output_dir.glob("checkpoint-*")
            if (step := _parse_checkpoint_step(path)) is not None
            and not is_archived(step)
        ),
        key=lambda item: item[0],
    )
    for _, path in regular_checkpoints[: max(0, len(regular_checkpoints) - keep_last)]:
        shutil.rmtree(path, ignore_errors=True)

    regular_weights = sorted(
        (
            (step, path)
            for path in output_dir.glob("s2mel_step*.pth")
            if (step := _weight_checkpoint_step(path)) is not None
            and not is_archived(step)
        ),
        key=lambda item: item[0],
    )
    for _, path in regular_weights[: max(0, len(regular_weights) - keep_last)]:
        path.unlink(missing_ok=True)


def _parse_checkpoint_step(path: Path) -> int | None:
    if not path.name.startswith("checkpoint-"):
        return None
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return None


def _read_compatible_checkpoint_metadata(path: Path) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    return int(state.get("epoch", 0)), int(state.get("iters", state.get("step", 0)))


def load_training_resume_state(resume_dir: Path) -> tuple[int, int, int]:
    """Recover (epoch, global_step, epoch_step) for an Accelerator checkpoint directory."""
    trainer_state = resume_dir / "trainer_state.json"
    if trainer_state.is_file():
        with trainer_state.open("r", encoding="utf-8") as f:
            state = json.load(f)
        return (
            int(state.get("epoch", 0)),
            int(state.get("global_step", 0)),
            int(state.get("epoch_step", 0)),
        )

    step = _parse_checkpoint_step(resume_dir)
    if step is None:
        return 0, 0, 0

    companion = resume_dir.parent / f"s2mel_step{step}.pth"
    if companion.is_file():
        epoch, iters = _read_compatible_checkpoint_metadata(companion)
        return epoch, iters, 0
    return 0, step, 0


def build_training_batch(
    batch: dict[str, Any],
    *,
    cfg,
    accelerator: Accelerator,
    feature_adapter_ref: list[IndexTTSFeatureAdapter | None],
) -> dict[str, torch.Tensor]:
    if batch.get("is_precomputed", False):
        return move_feature_batch_to_device(batch, accelerator.device)

    if feature_adapter_ref[0] is None:
        if accelerator.is_main_process:
            print("[Feature] Initializing frozen IndexTTS feature adapter")
        feature_adapter_ref[0] = IndexTTSFeatureAdapter(cfg).to(accelerator.device)
        feature_adapter_ref[0].eval()
    if batch.get("is_paired", False):
        return feature_adapter_ref[0].extract_paired_from_audio_paths(
            batch["prompt_audio_paths"],
            batch["target_audio_paths"],
            prompt_waveforms=batch.get("prompt_audio_waveforms"),
            prompt_sample_rates=batch.get("prompt_audio_sample_rates"),
            target_waveforms=batch.get("target_audio_waveforms"),
            target_sample_rates=batch.get("target_audio_sample_rates"),
        )
    if bool(_get(cfg.data, "random_split_audio", False)):
        return feature_adapter_ref[0].extract_random_split_from_audio_paths(
            batch["audio_paths"],
            waveforms=batch.get("audio_waveforms"),
            sample_rates=batch.get("audio_sample_rates"),
        )
    return feature_adapter_ref[0].extract_from_audio_paths(
        batch["audio_paths"],
        waveforms=batch.get("audio_waveforms"),
        sample_rates=batch.get("audio_sample_rates"),
    )


def forward_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    loss, _ = model(
        batch["mel"],
        batch["mel_lens"],
        batch["prompt_lens"],
        batch["semantic"],
        batch["style"],
        semantic_is_mu=False,
        semantic_lens=batch.get("semantic_lens"),
        prompt_semantic_lens=batch.get("prompt_semantic_lens"),
    )
    return loss


@torch.no_grad()
def validate(model, loader, cfg, accelerator: Accelerator, feature_adapter_ref) -> float:
    """Deterministic validation: fixed RNG so t / noise / prompt lengths are
    identical across evaluations, making valid/loss comparable over training."""
    model.eval()
    seed = int(cfg.seed)
    devices = [accelerator.device] if accelerator.device.type == "cuda" else []
    py_state = random.getstate()
    total = torch.zeros((), device=accelerator.device)
    count = torch.zeros((), device=accelerator.device)
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            random.seed(seed)
            for batch in loader:
                train_batch = build_training_batch(
                    batch,
                    cfg=cfg,
                    accelerator=accelerator,
                    feature_adapter_ref=feature_adapter_ref,
                )
                loss = forward_loss(model, train_batch)
                batch_size = train_batch["mel"].size(0)
                total = total + loss.detach() * batch_size
                count = count + batch_size
    finally:
        random.setstate(py_state)
        model.train()
    total = accelerator.reduce(total, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    if count.item() == 0:
        return float("nan")
    return (total / count).item()


def save_training_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    cfg,
    output_dir: Path,
    epoch: int,
    global_step: int,
    epoch_step: int = 0,
) -> None:
    save_dir = output_dir / f"checkpoint-{global_step}"
    accelerator.save_state(str(save_dir))
    if accelerator.is_main_process:
        with (save_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "epoch_step": int(epoch_step),
                },
                f,
                indent=2,
            )
        unwrapped = accelerator.unwrap_model(model)
        save_compatible_checkpoint(
            output_dir / f"s2mel_step{global_step}.pth",
            unwrapped,
            epoch=epoch,
            step=global_step,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        rotate_checkpoints(
            output_dir,
            int(cfg.train.keep_last),
            archive_interval=int(_get(cfg.train, "archive_save_interval", 0)),
        )
        print(f"[Checkpoint] Saved {save_dir}")


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(OmegaConf.load(args.config), args)
    train_source, train_is_speechdata = _split_source(cfg, "train")
    valid_source, valid_is_speechdata = _split_source(cfg, "valid")
    if not train_source:
        raise ValueError(
            "Set data.train_jsonl/data.train_speechdata_dir or pass "
            "--train-jsonl/--train-speechdata-dir"
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.train.grad_accumulation),
        mixed_precision=str(cfg.train.mixed_precision),
        log_with=None if bool(cfg.train.no_wandb) else "wandb",
    )
    # device_specific=True offsets the seed by the process index so DDP ranks
    # draw independent flow-matching timesteps / noise; the data-loader shuffle
    # generator is still synchronized across ranks by accelerate.
    set_seed(int(cfg.seed), device_specific=True)
    if not bool(cfg.train.no_wandb):
        wandb_project = str(_get(cfg.train, "wandb_project", "semantic2mel") or "semantic2mel")
        wandb_entity = str(_get(cfg.train, "wandb_entity", "") or "")
        wandb_run_name = str(_get(cfg.train, "wandb_run_name", "") or "")
        wandb_kwargs = {}
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        if wandb_run_name:
            wandb_kwargs["name"] = wandb_run_name
        tracker_kwargs = {"init_kwargs": {"wandb": wandb_kwargs}} if wandb_kwargs else {}
        accelerator.init_trackers(
            wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            **tracker_kwargs,
        )

    output_dir = Path(cfg.train.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.resolved.yaml")
    accelerator.wait_for_everyone()

    train_dataset = make_source_dataset(cfg, train_source, speechdata=train_is_speechdata)
    valid_dataset = (
        make_source_dataset(cfg, valid_source, speechdata=valid_is_speechdata) if valid_source else None
    )
    if bool(_get(cfg.data, "preload_features", False)):
        if accelerator.is_main_process:
            print("[Preload] Initializing frozen IndexTTS feature adapter")
        preload_adapter = IndexTTSFeatureAdapter(cfg).to(accelerator.device)
        preload_adapter.eval()
        train_dataset = preload_dataset_features(
            train_dataset,
            split="train",
            cfg=cfg,
            adapter=preload_adapter,
            accelerator=accelerator,
        )
        if valid_dataset is not None:
            valid_dataset = preload_dataset_features(
                valid_dataset,
                split="valid",
                cfg=cfg,
                adapter=preload_adapter,
                accelerator=accelerator,
            )
        del preload_adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    if bool(_get(cfg.data, "pair_same_speaker", True)):
        train_prompt_dataset = train_dataset
        valid_prompt_dataset = valid_dataset
        train_dataset = make_speaker_paired_dataset(cfg, train_prompt_dataset)
        if valid_prompt_dataset is not None:
            valid_dataset = make_speaker_paired_dataset(
                cfg,
                valid_prompt_dataset,
                fallback_prompt_dataset=train_prompt_dataset,
            )
        if accelerator.is_main_process:
            print(
                f"[Pairing] train: {len(train_dataset)} same-speaker targets; "
                f"missing speaker_id: {train_dataset.missing_speaker_count}"
            )
            if valid_dataset is not None:
                print(
                    f"[Pairing] valid: {len(valid_dataset)} same-speaker targets; "
                    f"train-prompt fallbacks: {valid_dataset.fallback_target_count}; "
                    f"missing speaker_id: {valid_dataset.missing_speaker_count}"
                )

    train_loader = make_dataloader(
        cfg,
        train_source,
        shuffle=True,
        speechdata=train_is_speechdata,
        dataset=train_dataset,
    )
    valid_loader = (
        make_dataloader(
            cfg,
            valid_source,
            shuffle=False,
            speechdata=valid_is_speechdata,
            persistent_workers=False,
            dataset=valid_dataset,
        )
        if valid_dataset is not None
        else None
    )

    model = Semantic2MelModel(cfg.s2mel)
    resume_from = str(cfg.train.resume_from or "")
    resume_path = Path(resume_from).expanduser() if resume_from else None
    validate_resume_backbone(cfg, resume_path)
    metadata = model_parameter_metadata(model, cfg)
    if accelerator.is_main_process:
        (output_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "[Model] "
            f"dit_type={metadata['dit_type']} "
            f"estimator_parameters={metadata['estimator_parameters']:,} "
            f"model_parameters={metadata['model_parameters']:,}"
        )
    accelerator.log(
        {
            "model/estimator_parameters": metadata["estimator_parameters"],
            "model/cfm_parameters": metadata["cfm_parameters"],
            "model/parameters": metadata["model_parameters"],
        },
        step=0,
    )
    start_epoch = 0
    global_step = 0
    resume_epoch_step = 0
    if resume_path is not None and resume_path.is_file():
        start_epoch, global_step = load_compatible_checkpoint(model, resume_path, strict=False)
        if accelerator.is_main_process:
            print(f"[Resume] Loaded compatible checkpoint {resume_path} at step={global_step}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )
    updates_per_epoch = math.ceil(len(train_loader) / int(cfg.train.grad_accumulation))
    total_steps = int(cfg.train.max_steps) if int(cfg.train.max_steps) > 0 else int(cfg.train.epochs) * updates_per_epoch
    scheduler = make_lr_scheduler(optimizer, cfg, num_training_steps=max(1, total_steps))
    if resume_path is not None and resume_path.is_file() and global_step > 0:
        # Weights-only checkpoints carry no scheduler state. Fast-forward the LR
        # schedule so training does not restart warmup at full LR. The prepared
        # scheduler ticks num_processes times per optimizer step, so replay the
        # equivalent number of raw ticks here (before accelerator.prepare).
        for _ in range(global_step * accelerator.num_processes):
            scheduler.step()
        if accelerator.is_main_process:
            print(
                f"[Resume] Fast-forwarded LR scheduler by {global_step} steps "
                f"(lr={scheduler.get_last_lr()[0]:.3e}); optimizer moments start fresh"
            )

    if valid_loader is None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    else:
        model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, valid_loader, scheduler
        )

    if resume_path is not None and resume_path.is_dir():
        start_epoch, global_step, resume_epoch_step = load_training_resume_state(resume_path)
        accelerator.load_state(str(resume_path))
        if args.resume_epoch_step is not None:
            resume_epoch_step = args.resume_epoch_step
        if accelerator.is_main_process:
            print(
                f"[Resume] Loaded accelerator state {resume_path} "
                f"at epoch={start_epoch + 1} step={global_step} epoch_step={resume_epoch_step}"
            )

    if _dit_type(cfg) == "DiT":
        accelerator.unwrap_model(model).models["cfm"].setup_estimator_caches(
            max_batch_size=int(cfg.train.batch_size),
            max_seq_length=int(cfg.s2mel.DiT.block_size),
        )

    feature_adapter_ref: list[IndexTTSFeatureAdapter | None] = [None]
    model.train()
    last_saved_step = global_step
    # Per-rank optimizer steps per epoch (prepared loader is already sharded).
    steps_per_epoch = max(1, math.ceil(len(train_loader) / int(cfg.train.grad_accumulation)))

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        epoch_step = 0
        epoch_loader = train_loader
        if epoch == start_epoch and resume_epoch_step > 0:
            skip_batches = resume_epoch_step * int(cfg.train.grad_accumulation)
            epoch_loader = accelerator.skip_first_batches(train_loader, skip_batches)
            epoch_step = resume_epoch_step
            if accelerator.is_main_process:
                print(f"[Resume] Skipping first {skip_batches} batches of epoch {epoch + 1}")
        for raw_batch in epoch_loader:
            if int(cfg.train.max_steps) > 0 and global_step >= int(cfg.train.max_steps):
                break
            with accelerator.accumulate(model):
                train_batch = build_training_batch(
                    raw_batch,
                    cfg=cfg,
                    accelerator=accelerator,
                    feature_adapter_ref=feature_adapter_ref,
                )
                loss = forward_loss(model, train_batch)
                accelerator.backward(loss)
                if accelerator.sync_gradients and float(cfg.train.grad_clip) > 0:
                    accelerator.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                epoch_step += 1
                if global_step % int(cfg.train.log_interval) == 0:
                    reduced_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
                    lr = scheduler.get_last_lr()[0]
                    # Fractional completed epochs, e.g. 1.0 == first epoch done.
                    epoch_progress = epoch + min(1.0, epoch_step / steps_per_epoch)
                    if accelerator.is_main_process:
                        print(f"[Train] epoch={epoch + 1} step={global_step} loss={reduced_loss:.5f} lr={lr:.3e}")
                    accelerator.log(
                        {"train/loss": reduced_loss, "train/lr": lr, "train/epoch": epoch_progress},
                        step=global_step,
                    )

                if valid_loader is not None and global_step % int(cfg.train.valid_interval) == 0:
                    val_loss = validate(model, valid_loader, cfg, accelerator, feature_adapter_ref)
                    if accelerator.is_main_process:
                        print(f"[Valid] step={global_step} loss={val_loss:.5f}")
                    accelerator.log({"valid/loss": val_loss}, step=global_step)

                save_interval = int(cfg.train.save_interval)
                archive_interval = int(_get(cfg.train, "archive_save_interval", 0))
                save_regular = save_interval > 0 and global_step % save_interval == 0
                save_archive = archive_interval > 0 and global_step % archive_interval == 0
                if save_regular or save_archive:
                    save_training_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        cfg=cfg,
                        output_dir=output_dir,
                        epoch=epoch,
                        global_step=global_step,
                        epoch_step=epoch_step,
                    )
                    last_saved_step = global_step

        if int(cfg.train.max_steps) > 0 and global_step >= int(cfg.train.max_steps):
            break

    if global_step > 0 and last_saved_step != global_step:
        save_training_checkpoint(
            accelerator=accelerator,
            model=model,
            cfg=cfg,
            output_dir=output_dir,
            epoch=int(cfg.train.epochs),
            global_step=global_step,
        )
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_compatible_checkpoint(
            output_dir / "s2mel_final.pth",
            unwrapped,
            epoch=int(cfg.train.epochs),
            step=global_step,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"[Done] Finished at step={global_step}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
