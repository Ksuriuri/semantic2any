from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import sys
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from semantic2any.data.s2mel_dataset import (
    DEFAULT_MAX_AUDIO_SECONDS,
    DEFAULT_MAX_PAIR_SECONDS,
    DEFAULT_MAX_PROMPT_SECONDS,
    LengthBucketBatchSampler,
    S2MelCollator,
    S2MelInMemoryDataset,
    S2MelJsonlDataset,
    S2MelSpeakerPairedDataset,
    S2MelSpeechDataDataset,
)
from semantic2any.models import Semantic2MelModel
from semantic2any.utils.checkpoint import load_compatible_checkpoint, save_compatible_checkpoint
from semantic2any.utils.indextts_adapters import (
    S2MelFeatureAdapter,
    build_feature_adapter,
    move_feature_batch_to_device,
)
from semantic2any.utils.semantic_codecs import (
    resolve_semantic_codec_config,
    semantic_codec_info,
    semantic_codec_type,
)


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


def model_parameter_metadata(model, cfg) -> dict[str, int | float | str]:
    cfm = model.models["cfm"]
    codec = semantic_codec_info(cfg)
    return {
        "dit_type": _dit_type(cfg),
        "semantic_codec": codec.name,
        "semantic_source_model": codec.source_model,
        "semantic_dim": codec.semantic_dim,
        "semantic_fps": codec.semantic_fps,
        "semantic_fingerprint": codec.fingerprint(),
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
    checkpoint_codec = semantic_codec_type(checkpoint_config)
    current_codec = semantic_codec_type(cfg)
    if checkpoint_codec != current_codec:
        raise ValueError(
            f"Cannot resume {checkpoint_codec} semantic checkpoint with {current_codec} "
            "config. Semantic codec checkpoints are not compatible."
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
    parser.add_argument("--model-dir", default=None)
    parser.add_argument(
        "--semantic-codec",
        choices=("maskgct", "sac"),
        default=None,
        help="Override the semantic feature backend.",
    )
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
    resolve_semantic_codec_config(cfg, args.semantic_codec)
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


def make_source_dataset(
    cfg,
    source: str,
    *,
    speechdata: bool = False,
    rank: int | None = None,
    world_size: int = 1,
) -> Dataset:
    if speechdata:
        return S2MelSpeechDataDataset(
            source,
            cache_dir=_get(cfg.data, "speechdata_cache_dir", None),
        )
    return S2MelJsonlDataset(
        source,
        rank=rank,
        world_size=world_size,
        drop_unused_fields=rank is not None,
    )


def _dataset_length_estimates(dataset: Dataset) -> list[float]:
    estimate_fn = getattr(dataset, "estimated_sample_seconds", None)
    if callable(estimate_fn):
        return [float(estimate_fn(index)) for index in range(len(dataset))]
    records = getattr(dataset, "records", None)
    if isinstance(records, list) and len(records) == len(dataset):
        lengths = []
        for record in records:
            duration = record.get("duration") if isinstance(record, dict) else None
            lengths.append(float(duration) if isinstance(duration, (int, float)) else 0.0)
        return lengths
    return [0.0] * len(dataset)




def _compute_dataset_sample_weights(dataset: Dataset, dataset_weights: dict[str, float]) -> list[float] | None:
    """Return per-sample weights derived from dataset_weights config, or None if not configured."""
    if not dataset_weights:
        return None
    records = getattr(dataset, "records", None)
    if not isinstance(records, list) or len(records) != len(dataset):
        return None
    weights = []
    for record in records:
        dataset_name = _record_dataset_name(record) if isinstance(record, dict) else ""
        # Match by prefix: config key "laion_emolia" matches "laion_emolia__ZH_..."
        w = 1.0
        for key, val in dataset_weights.items():
            if dataset_name == key or dataset_name.startswith(key + "__") or dataset_name.startswith(key + "/"):
                w = float(val)
                break
        weights.append(w)
    return weights
def _set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    seen: set[int] = set()

    def visit(obj) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        setter = getattr(obj, "set_epoch", None)
        if callable(setter):
            setter(epoch)
        visit(getattr(obj, "batch_sampler", None))
        visit(getattr(obj, "sampler", None))

    visit(loader)


def make_dataloader(
    cfg,
    source: str,
    shuffle: bool,
    *,
    speechdata: bool = False,
    persistent_workers: bool = True,
    dataset: Dataset | None = None,
    world_size: int = 1,
) -> DataLoader:
    if dataset is None:
        dataset = make_source_dataset(cfg, source, speechdata=speechdata)
    spect = cfg.preprocess_params.spect_params
    codec = semantic_codec_info(cfg)
    mel_fmax = _get(spect, "fmax", "None")
    mel_fmax = None if mel_fmax in (None, "None") else float(mel_fmax)
    collator = S2MelCollator(
        hop_length=int(spect.hop_length),
        sample_rate=int(cfg.preprocess_params.sr),
        min_prompt_seconds=float(cfg.data.min_prompt_seconds),
        max_prompt_seconds=_optional_float(
            _get(cfg.data, "max_prompt_seconds", DEFAULT_MAX_PROMPT_SECONDS)
        ),
        min_generated_frames=int(cfg.data.min_generated_frames),
        min_target_seconds=_optional_float(_get(cfg.data, "min_target_seconds", None)),
        max_target_seconds=_optional_float(_get(cfg.data, "max_target_seconds", None)),
        max_pair_seconds=float(
            _get(cfg.data, "max_pair_seconds", DEFAULT_MAX_PAIR_SECONDS)
        ),
        min_pair_prompt_seconds=float(_get(cfg.data, "min_pair_prompt_seconds", 3.0)),
        decode_audio_in_worker=bool(_get(cfg.data, "decode_audio_in_worker", False)),
        skip_audio_errors=bool(_get(cfg.data, "skip_audio_errors", False)),
        max_audio_seconds=_optional_float(
            _get(cfg.data, "max_audio_seconds", DEFAULT_MAX_AUDIO_SECONDS)
        ),
        expected_semantic_codec=codec.name,
        expected_semantic_fingerprint=codec.fingerprint(),
        extract_mel_in_worker=bool(_get(cfg.data, "extract_mel_in_worker", False)),
        mel_n_fft=int(_get(spect, "n_fft", 2048)),
        mel_win_length=int(_get(spect, "win_length", 2048)),
        mel_n_mels=int(_get(spect, "n_mels", 128)),
        mel_fmin=float(_get(spect, "fmin", 0.0)),
        mel_fmax=mel_fmax,
        prompt_bandwidth_aug_prob=float(
            _get(cfg.data, "prompt_bandwidth_aug_prob", 0.3)
        ),
        prompt_bandwidth_aug_rates=tuple(
            int(rate)
            for rate in _get(cfg.data, "prompt_bandwidth_aug_rates", (16000, 22050))
        ),
        force_bandwidth_hz=int(_get(cfg.data, "force_bandwidth_hz", 0) or 0),
    )
    kwargs: dict[str, Any] = {}
    if int(cfg.data.num_workers) > 0:
        kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
        # Validation loaders re-create workers per pass so that the seeded RNG
        # fork in validate() also controls worker seeding (deterministic prompts).
        kwargs["persistent_workers"] = persistent_workers
        # If a worker wedges on a bad tar/flac, fail this rank in 180s so it
        # can still join the finite-loss allreduce instead of dying at NCCL 600s.
        kwargs["timeout"] = 180.0
    use_buckets = bool(_get(cfg.data, "length_bucketed_batches", False)) and shuffle
    if use_buckets:
        boundaries = [
            float(value)
            for value in _get(cfg.data, "length_bucket_boundaries", (8, 12, 16, 20, 24, 28, 32, 40, 50))
        ]
        _raw_dataset_weights = _get(cfg.data, "dataset_weights", {}) or {}
        _sample_weights = _compute_dataset_sample_weights(dataset, dict(_raw_dataset_weights)) if shuffle else None
        batch_sampler = LengthBucketBatchSampler(
            _dataset_length_estimates(dataset),
            batch_size=int(cfg.train.batch_size),
            world_size=int(world_size),
            boundaries=boundaries,
            seed=int(cfg.seed),
            drop_last=True,
            shuffle=True,
            sample_weights=_sample_weights,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=int(cfg.data.num_workers),
            collate_fn=collator,
            pin_memory=bool(_get(cfg.data, "pin_memory", True)),
            **kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        shuffle=shuffle,
        num_workers=int(cfg.data.num_workers),
        collate_fn=collator,
        pin_memory=bool(_get(cfg.data, "pin_memory", True)),
        drop_last=shuffle,
        **kwargs,
    )


def _sync_has_batch(accelerator: Accelerator, has_batch: bool) -> bool:
    """Agree across ranks on whether the epoch continues.

    Shards are split by speaker hash, so they are not exactly equal: the
    smallest runs out ~1.6% of an epoch before the largest.  A rank that leaves
    the epoch loop on its own never joins the next step's collectives, and the
    others block in the numel=1 finite-loss all-reduce until the 600 s NCCL
    watchdog aborts the job.  Reducing with MIN ends the epoch for everyone as
    soon as any rank is out, at the cost of the largest shard's tail.
    """
    if accelerator.num_processes <= 1:
        return has_batch
    flag = torch.tensor([1.0 if has_batch else 0.0], device=accelerator.device)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    out_of_data = flag.item() < 1.0
    if has_batch and out_of_data and accelerator.is_main_process:
        print(
            "[EpochEnd] another rank ran out of data - all ranks end the epoch "
            "together",
            flush=True,
        )
    return not out_of_data


def _next_raw_batch(iterator):
    # Return (batch, exhausted). Fetch errors return (None, False).
    try:
        return next(iterator), False
    except StopIteration:
        return None, True
    except Exception as exc:
        print(f"[DataLoader] fetch failed: {type(exc).__name__}: {exc}", flush=True)
        return None, False


_BUILD_POOL: ThreadPoolExecutor | None = None


def _build_batch_or_none(build_fn, raw_batch):
    # Same 180s budget as DataLoader timeout. A wedged build must still
    # reach the finite-loss allreduce instead of dying at NCCL 600s.
    global _BUILD_POOL
    if raw_batch is None:
        return None
    if _BUILD_POOL is None:
        _BUILD_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s2mel-build")
    fut = _BUILD_POOL.submit(build_fn, raw_batch)
    try:
        return fut.result(timeout=180.0)
    except Exception as exc:
        print(f"[DataLoader] build failed: {type(exc).__name__}: {exc}", flush=True)
        _BUILD_POOL.shutdown(wait=False, cancel_futures=True)
        _BUILD_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s2mel-build")
        return None


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
    adapter: S2MelFeatureAdapter,
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
        if any(record.get("semantic_code_path") for record in source_records):
            raise ValueError(
                "data.preload_features is not supported with precomputed semantic codes; "
                "leave preload_features=false"
            )
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
) -> S2MelSpeakerPairedDataset:
    spect = cfg.preprocess_params.spect_params
    min_target_seconds = _optional_float(_get(cfg.data, "min_target_seconds", None))
    max_prompt_seconds = _optional_float(
        _get(cfg.data, "max_prompt_seconds", DEFAULT_MAX_PROMPT_SECONDS)
    )
    max_target_seconds = _optional_float(_get(cfg.data, "max_target_seconds", None))
    if min_target_seconds is None:
        min_target_seconds = (
            int(_get(cfg.data, "min_generated_frames", 8))
            * int(spect.hop_length)
            / int(cfg.preprocess_params.sr)
        )
    if max_prompt_seconds is None:
        max_prompt_seconds = float(
            _get(cfg.data, "max_audio_seconds", DEFAULT_MAX_AUDIO_SECONDS)
        )
    if max_target_seconds is None:
        max_target_seconds = float(
            _get(cfg.data, "max_audio_seconds", DEFAULT_MAX_AUDIO_SECONDS)
        )
    return S2MelSpeakerPairedDataset(
        dataset,
        min_prompt_seconds=float(_get(cfg.data, "min_pair_prompt_seconds", 3.0)),
        max_prompt_seconds=max_prompt_seconds,
        min_target_seconds=min_target_seconds,
        max_target_seconds=max_target_seconds,
        hop_length=int(spect.hop_length),
        sample_rate=int(cfg.preprocess_params.sr),
        seed=int(cfg.seed),
        allow_singleton_split=bool(
            _get(cfg.data, "allow_singleton_split", True)
        ),
    )


def _weight_checkpoint_step(
    path: Path, prefix: str = "s2mel_step", suffix: str = ".pth"
) -> int | None:
    name = path.name
    if not name.startswith(prefix) or not name.endswith(suffix):
        return None
    try:
        return int(name.removeprefix(prefix).removesuffix(suffix))
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

    # Joint training writes a 489 MiB vocoder beside every s2mel_step*.pth; left
    # unrotated that fills the disk faster than the model weights do.  Same
    # keep_last / archive policy so a step that keeps its model keeps its vocoder.
    regular_vocoders = sorted(
        (
            (step, path)
            for path in output_dir.glob("bigvgan_step*.pt")
            if (step := _weight_checkpoint_step(path, "bigvgan_step", ".pt")) is not None
            and not is_archived(step)
        ),
        key=lambda item: item[0],
    )
    for _, path in regular_vocoders[: max(0, len(regular_vocoders) - keep_last)]:
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
    feature_adapter_ref: list[S2MelFeatureAdapter | None],
    apply_prompt_bandwidth_aug: bool = True,
) -> dict[str, torch.Tensor]:
    built = _build_training_batch_impl(
        batch,
        cfg=cfg,
        accelerator=accelerator,
        feature_adapter_ref=feature_adapter_ref,
        apply_prompt_bandwidth_aug=apply_prompt_bandwidth_aug,
    )
    # Re-attach per-sample identifiers dropped by the feature adapters so the
    # spike-sample instrumentation can name the offending audio.
    try:
        if isinstance(batch, dict) and isinstance(built, dict):
            for _k in ("target_audio_paths", "audio_paths", "prompt_audio_paths", "records"):
                _v = batch.get(_k)
                if _v is not None and _k not in built:
                    built[_k] = _v
    except Exception:
        pass
    return built


def _build_training_batch_impl(
    batch: dict[str, Any],
    *,
    cfg,
    accelerator: Accelerator,
    feature_adapter_ref: list[S2MelFeatureAdapter | None],
    apply_prompt_bandwidth_aug: bool = True,
) -> dict[str, torch.Tensor]:
    if batch.get("is_precomputed", False):
        return move_feature_batch_to_device(batch, accelerator.device)

    has_semantic_codes = bool(batch.get("has_semantic_codes", False))
    if feature_adapter_ref[0] is None:
        if accelerator.is_main_process:
            codec = semantic_codec_info(cfg)
            if has_semantic_codes:
                print(
                    f"[Feature] Initializing {codec.name} code lookup adapter "
                    f"({codec.semantic_fps:g} Hz, {codec.semantic_dim} dims)"
                )
            else:
                print(
                    f"[Feature] Initializing {codec.name} semantic adapter "
                    f"({codec.semantic_fps:g} Hz, {codec.semantic_dim} dims)"
                )
        feature_adapter_ref[0] = build_feature_adapter(
            cfg,
            semantic_lookup_path=(
                batch["semantic_lookup_path"] if has_semantic_codes else None
            ),
            semantic_lookup_sha256=(
                batch["semantic_lookup_sha256"] if has_semantic_codes else None
            ),
        ).to(accelerator.device)
        feature_adapter_ref[0].eval()
    elif has_semantic_codes:
        decoder = feature_adapter_ref[0].semantic_decoder
        if decoder is None:
            raise ValueError(
                "Cannot mix precomputed semantic-code batches with online semantic batches"
            )
        if decoder.lookup_sha256 != batch["semantic_lookup_sha256"]:
            raise ValueError(
                "All training and validation batches must use the same lookup table"
            )
    elif feature_adapter_ref[0].semantic_backend is None:
        raise ValueError(
            "Cannot mix online semantic batches with precomputed semantic-code batches"
        )
    if batch.get("worker_precomputed_mel", False):
        return feature_adapter_ref[0].finalize_worker_paired_batch(batch)
    if batch.get("is_paired", False):
        return feature_adapter_ref[0].extract_paired_from_audio_paths(
            batch["prompt_audio_paths"],
            batch["target_audio_paths"],
            prompt_waveforms=batch.get("prompt_audio_waveforms"),
            prompt_sample_rates=batch.get("prompt_audio_sample_rates"),
            target_waveforms=batch.get("target_audio_waveforms"),
            target_sample_rates=batch.get("target_audio_sample_rates"),
            singleton_splits=batch.get("singleton_splits"),
            prompt_semantic_codes=batch.get("prompt_semantic_codes"),
            prompt_semantic_code_lens=batch.get("prompt_semantic_code_lens"),
            target_semantic_codes=batch.get("target_semantic_codes"),
            target_semantic_code_lens=batch.get("target_semantic_code_lens"),
            apply_prompt_bandwidth_aug=apply_prompt_bandwidth_aug,
        )
    if bool(_get(cfg.data, "random_split_audio", False)):
        return feature_adapter_ref[0].extract_random_split_from_audio_paths(
            batch["audio_paths"],
            waveforms=batch.get("audio_waveforms"),
            sample_rates=batch.get("audio_sample_rates"),
            semantic_codes=batch.get("semantic_codes"),
            semantic_code_lens=batch.get("semantic_code_lens"),
            apply_prompt_bandwidth_aug=apply_prompt_bandwidth_aug,
        )
    return feature_adapter_ref[0].extract_from_audio_paths(
        batch["audio_paths"],
        waveforms=batch.get("audio_waveforms"),
        sample_rates=batch.get("audio_sample_rates"),
        semantic_codes=batch.get("semantic_codes"),
        semantic_code_lens=batch.get("semantic_code_lens"),
    )


def _record_tensors_on_stream(value: Any, stream: torch.cuda.Stream) -> None:
    """Keep nested CUDA tensors alive until work on ``stream`` completes."""
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _record_tensors_on_stream(item, stream)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _record_tensors_on_stream(item, stream)


class AsyncFeatureBatchBuilder:
    """Build one training batch ahead on a dedicated thread and CUDA stream."""

    def __init__(
        self,
        build_fn: Callable[[dict[str, Any]], dict[str, torch.Tensor]],
        *,
        device: torch.device,
    ) -> None:
        self.build_fn = build_fn
        self.device = torch.device(device)
        self.feature_stream = (
            torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        )
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="s2mel-feature")
        self.future: Future[
            tuple[dict[str, torch.Tensor], torch.cuda.Event | None]
        ] | None = None
        self.closed = False

    @property
    def has_pending(self) -> bool:
        return self.future is not None

    def submit(self, raw_batch: dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("Cannot submit to a closed feature batch builder")
        if self.has_pending:
            raise RuntimeError("Only one feature batch may be in flight")

        input_event = None
        if self.feature_stream is not None:
            with torch.cuda.device(self.device):
                input_event = torch.cuda.Event()
                input_event.record(torch.cuda.current_stream(self.device))
        self.future = self.executor.submit(self._build, raw_batch, input_event)

    def _build(
        self,
        raw_batch: dict[str, Any],
        input_event: torch.cuda.Event | None,
    ) -> tuple[dict[str, torch.Tensor], torch.cuda.Event | None]:
        if self.feature_stream is None:
            with torch.no_grad():
                return self.build_fn(raw_batch), None

        with torch.cuda.device(self.device), torch.cuda.stream(self.feature_stream):
            if input_event is not None:
                self.feature_stream.wait_event(input_event)
            _record_tensors_on_stream(raw_batch, self.feature_stream)
            # Keep outputs as normal tensors: the trainable model may save
            # semantic/style inputs for parameter-gradient computation.
            with torch.no_grad():
                batch = self.build_fn(raw_batch)
            ready_event = torch.cuda.Event()
            ready_event.record(self.feature_stream)
        return batch, ready_event

    def _take_result(
        self,
    ) -> tuple[dict[str, torch.Tensor], torch.cuda.Event | None]:
        if self.future is None:
            raise RuntimeError("No feature batch is pending")
        future = self.future
        self.future = None
        return future.result()

    def _prepare_for_consumer(
        self,
        result: tuple[dict[str, torch.Tensor], torch.cuda.Event | None],
    ) -> dict[str, torch.Tensor]:
        batch, ready_event = result
        if ready_event is not None:
            with torch.cuda.device(self.device):
                consumer_stream = torch.cuda.current_stream(self.device)
                consumer_stream.wait_event(ready_event)
                _record_tensors_on_stream(batch, consumer_stream)
        return batch

    def get(self, timeout: float = 180.0) -> dict[str, torch.Tensor]:
        if self.future is None:
            raise RuntimeError("No feature batch is pending")
        future = self.future
        self.future = None
        try:
            result = future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            raise TimeoutError("feature batch build timed out") from exc
        return self._prepare_for_consumer(result)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        pending = None
        if self.future is not None:
            future = self.future
            self.future = None
            try:
                pending = future.result()
            except BaseException:
                pending = None
        self.executor.shutdown(wait=True, cancel_futures=True)
        if pending is not None and pending[1] is not None:
            pending[1].synchronize()
        if self.feature_stream is not None:
            self.feature_stream.synchronize()


def async_feature_extraction_enabled(cfg, accelerator: Accelerator) -> bool:
    return (
        bool(_get(cfg.data, "async_feature_extraction", False))
        and not bool(_get(cfg.data, "preload_features", False))
        and accelerator.device.type == "cuda"
    )


def step_requires_async_prefetch_barrier(
    cfg,
    *,
    sync_gradients: bool,
    next_global_step: int,
    has_validation: bool,
) -> bool:
    """Avoid prefetching across stateful maintenance boundaries."""
    if not sync_gradients:
        return False
    valid_interval = int(cfg.train.valid_interval)
    save_interval = int(cfg.train.save_interval)
    archive_interval = int(_get(cfg.train, "archive_save_interval", 0))
    max_steps = int(cfg.train.max_steps)
    return (
        (has_validation and valid_interval > 0 and next_global_step % valid_interval == 0)
        or (save_interval > 0 and next_global_step % save_interval == 0)
        or (archive_interval > 0 and next_global_step % archive_interval == 0)
        or (max_steps > 0 and next_global_step >= max_steps)
    )



# --- spike-sample instrumentation (22kHz loss-spike debug) ---------------------
import os as _spike_os

_SPIKE_LOSS_ABS = float(_spike_os.environ.get("SPIKE_LOSS_ABS", "1.5"))
_SPIKE_LOSS_MULT = float(_spike_os.environ.get("SPIKE_LOSS_MULT", "2.5"))
_SPIKE_GNORM_ABS = float(_spike_os.environ.get("SPIKE_GNORM_ABS", "10.0"))
_SPIKE_GNORM_MULT = float(_spike_os.environ.get("SPIKE_GNORM_MULT", "4.0"))
_SPIKE_LOSS_EMA = [None]
_SPIKE_GNORM_EMA = [None]


def _spike_ids_from_batch(batch: dict) -> list:
    """Best-effort per-sample identifiers for the current micro-batch."""
    for key in ("target_audio_paths", "audio_paths", "prompt_audio_paths"):
        v = batch.get(key)
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return list(v)
    recs = batch.get("records")
    if isinstance(recs, (list, tuple)) and len(recs) > 0:
        out = []
        for r in recs:
            if isinstance(r, dict):
                out.append(r.get("audio_path") or r.get("id") or "?")
            else:
                out.append(str(r))
        return out
    return []


def _spike_track_loss(loss, batch, global_step, accelerator) -> None:
    """Log micro-batch sample ids whenever this rank's local loss spikes."""
    try:
        lv = float(loss.detach().float().item())
    except Exception:
        return
    ema = _SPIKE_LOSS_EMA[0]
    thr = max(_SPIKE_LOSS_ABS, _SPIKE_LOSS_MULT * ema) if ema is not None else _SPIKE_LOSS_ABS
    if (not math.isfinite(lv)) or lv > thr:
        ids = _spike_ids_from_batch(batch)
        ema_s = "None" if ema is None else f"{ema:.4f}"
        print(
            f"[SpikeSample] step~{global_step} rank={accelerator.process_index} "
            f"loss={lv:.4f} ema={ema_s} thr={thr:.4f} n={len(ids)} ids={ids}",
            flush=True,
        )
    if math.isfinite(lv):
        _SPIKE_LOSS_EMA[0] = lv if ema is None else (0.98 * ema + 0.02 * lv)


def _spike_track_gnorm(gn, global_step, accelerator) -> None:
    """Log grad-norm spikes (pre-clip total norm from clip_grad_norm_)."""
    try:
        gnf = float(gn) if gn is not None else float("nan")
    except Exception:
        return
    ema = _SPIKE_GNORM_EMA[0]
    thr = max(_SPIKE_GNORM_ABS, _SPIKE_GNORM_MULT * ema) if ema is not None else _SPIKE_GNORM_ABS
    if (not math.isfinite(gnf)) or gnf > thr:
        ema_s = "None" if ema is None else f"{ema:.3f}"
        print(
            f"[SpikeGrad] step~{global_step} rank={accelerator.process_index} "
            f"gnorm={gnf:.3f} ema={ema_s} thr={thr:.3f}",
            flush=True,
        )
    if math.isfinite(gnf):
        _SPIKE_GNORM_EMA[0] = gnf if ema is None else (0.98 * ema + 0.02 * gnf)
_SPIKE_SKIP_ENABLE = _spike_os.environ.get("SPIKE_SKIP_ENABLE", "0") == "1"
_SPIKE_SKIP_GNORM = float(_spike_os.environ.get("SPIKE_SKIP_GNORM", "15.0"))


def _spike_should_skip_step(gn) -> bool:
    """True when this optimizer step should be skipped (grad explosion / NaN)."""
    if not _SPIKE_SKIP_ENABLE:
        return False
    try:
        gnf = float(gn) if gn is not None else float("nan")
    except Exception:
        return False
    if not math.isfinite(gnf):
        return True
    return gnf > _SPIKE_SKIP_GNORM


# --- end instrumentation -------------------------------------------------------

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
                    apply_prompt_bandwidth_aug=False,
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
        if _VOCODER_GAN is not None:
            # accelerator.save_state only writes what was passed to prepare(),
            # and the vocoder deliberately is not.  Without this the resumed run
            # would pair an adapted flow model with a pristine HF vocoder, and
            # every listening pack would be rendered with the wrong vocoder.
            torch.save(_VOCODER_GAN.training_state(), save_dir / "vocoder_gan.pt")
            torch.save(
                {
                    "vocoder": _VOCODER_GAN.vocoder.state_dict(),
                    "step": int(global_step),
                    "weight_norm": True,
                },
                output_dir / f"bigvgan_step{global_step}.pt",
            )
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
            epoch_step=epoch_step,
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

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(_get(cfg.train, "find_unused_parameters", False))
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.train.grad_accumulation),
        mixed_precision=str(cfg.train.mixed_precision),
        log_with=None if bool(cfg.train.no_wandb) else "wandb",
        kwargs_handlers=[ddp_kwargs],
    )
    # device_specific=True offsets the seed by the process index so DDP ranks
    # draw independent flow-matching timesteps / noise; the data-loader shuffle
    # generator is still synchronized across ranks by accelerate.
    set_seed(int(cfg.seed), device_specific=True)
    if not bool(cfg.train.no_wandb):
        wandb_project = str(_get(cfg.train, "wandb_project", "semantic2mel") or "semantic2mel")
        wandb_entity = str(_get(cfg.train, "wandb_entity", "") or "")
        wandb_run_name = str(_get(cfg.train, "wandb_run_name", "") or "")
        wandb_run_id = str(_get(cfg.train, "wandb_run_id", "") or "")
        wandb_kwargs = {}
        if wandb_entity:
            wandb_kwargs["entity"] = wandb_entity
        if wandb_run_name:
            wandb_kwargs["name"] = wandb_run_name
        if wandb_run_id:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"
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

    train_dataset = make_source_dataset(
        cfg,
        train_source,
        speechdata=train_is_speechdata,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    valid_dataset = (
        make_source_dataset(cfg, valid_source, speechdata=valid_is_speechdata) if valid_source else None
    )
    if bool(_get(cfg.data, "preload_features", False)):
        if accelerator.is_main_process:
            print(
                f"[Preload] Initializing frozen {semantic_codec_type(cfg)} feature adapter"
            )
        preload_adapter = build_feature_adapter(cfg).to(accelerator.device)
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
            valid_dataset = make_speaker_paired_dataset(cfg, valid_prompt_dataset)
        if accelerator.is_main_process:
            print(
                f"[Pairing] train: {len(train_dataset)} samples "
                f"({train_dataset.paired_target_count} paired, "
                f"{train_dataset.singleton_target_count} singleton); "
                f"skipped target too-short={train_dataset.too_short_target_count}, "
                f"overlong={train_dataset.overlong_target_count}, "
                f"unusable={train_dataset.unusable_target_count}, "
                f"no-reference={train_dataset.singleton_dropped_count}, "
                f"missing-speaker={train_dataset.missing_speaker_count}, "
                f"missing-duration={train_dataset.missing_duration_count}"
            )
            if valid_dataset is not None:
                print(
                    f"[Pairing] valid: {len(valid_dataset)} samples "
                    f"({valid_dataset.paired_target_count} paired, "
                    f"{valid_dataset.singleton_target_count} singleton); "
                    f"no-reference={valid_dataset.singleton_dropped_count}, "
                    f"missing-speaker={valid_dataset.missing_speaker_count}, "
                    f"missing-duration={valid_dataset.missing_duration_count}"
                )

    train_loader = make_dataloader(
        cfg,
        train_source,
        shuffle=True,
        speechdata=train_is_speechdata,
        dataset=train_dataset,
        # Each rank already holds a disjoint speaker-group shard of the data,
        # so the local sampler must NOT be split across ranks again.
        world_size=1,
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
            f"semantic_codec={metadata['semantic_codec']} "
            f"semantic_dim={metadata['semantic_dim']} "
            f"semantic_fps={metadata['semantic_fps']:g} "
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
    resume_meta = None
    if resume_path is not None and resume_path.is_file():
        start_epoch, global_step, resume_meta = load_compatible_checkpoint(
            model, resume_path, strict=False, return_meta=True
        )
        stored_epoch_step = resume_meta.get("epoch_step")
        if stored_epoch_step is not None:
            resume_epoch_step = int(stored_epoch_step)
        if args.resume_epoch_step is not None:
            resume_epoch_step = args.resume_epoch_step
        if accelerator.is_main_process:
            print(f"[Resume] Loaded compatible checkpoint {resume_path} at step={global_step}")

    _fresh_lr = bool(_get(cfg.train, "fresh_lr_schedule", False))
    _sched_src = cfg
    if (
        resume_path is not None
        and resume_path.is_file()
        and not _fresh_lr
        and resume_meta is not None
        and resume_meta.get("config") is not None
    ):
        _sched_src = resume_meta["config"]
    _sched_train = _get(_sched_src, "train", _sched_src)
    _base_lr = float(_get(_sched_train, "learning_rate", cfg.train.learning_rate))
    _min_lr = float(_get(_sched_train, "min_learning_rate", _get(cfg.train, "min_learning_rate", 1.0e-5)))
    _warmup_steps = int(_get(_sched_train, "warmup_steps", cfg.train.warmup_steps))
    _max_steps_sched = int(_get(_sched_train, "max_steps", cfg.train.max_steps))
    if _base_lr <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {_base_lr}")
    if not 0.0 <= _min_lr <= _base_lr:
        raise ValueError(
            f"min_learning_rate must be between 0 and learning_rate; got {_min_lr} and {_base_lr}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=_base_lr,
        weight_decay=float(cfg.train.weight_decay),
    )
    updates_per_epoch = math.ceil(len(train_loader) / int(cfg.train.grad_accumulation))
    total_steps = int(_max_steps_sched) if int(_max_steps_sched) > 0 else int(cfg.train.epochs) * updates_per_epoch
    # AcceleratedScheduler ticks the LR schedule num_processes times per optimizer step
    # (when split_batches=False, which is the default). Scale num_training_steps and
    # warmup_steps accordingly so the cosine period matches the intended optimizer steps.
    _dl_cfg = getattr(accelerator, "dataloader_config", None)
    _split = getattr(_dl_cfg, "split_batches", False) or getattr(accelerator, "split_batches", False)
    _sched_scale = 1 if _split else accelerator.num_processes
    _warmup_scaled = int(_warmup_steps) * _sched_scale
    _total_scaled = max(1, total_steps * _sched_scale)
    scheduler = cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=_warmup_scaled,
        num_training_steps=_total_scaled,
        min_lr_ratio=_min_lr / _base_lr,
    )
    if resume_path is not None and resume_path.is_file() and global_step > 0:
        if _fresh_lr:
            # New schedule from the live yaml. Keep epoch / epoch_step so already-seen
            # samples are still skipped; pass --resume-epoch-step 0 to replay them.
            if accelerator.is_main_process:
                print(
                    f"[Resume] fresh_lr_schedule=True: LR starts at {float(cfg.train.learning_rate):.2e}, "
                    f"epoch_step={resume_epoch_step} kept"
                )
        else:
            # Weights-only checkpoints carry no scheduler state. Rebuild from the
            # checkpoint's own train hparams (not a later yaml edit) and fast-forward.
            for _ in range(global_step * _sched_scale):
                scheduler.step()
            if accelerator.is_main_process:
                print(
                    f"[Resume] Restored LR from checkpoint schedule "
                    f"(base={_base_lr:.3e}, min={_min_lr:.3e}, warmup={_warmup_steps}) "
                    f"fast-forward {global_step} steps -> lr={scheduler.get_last_lr()[0]:.3e}; "
                    "optimizer moments start fresh"
                )

    if valid_loader is None:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    else:
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        valid_loader = accelerator.prepare(valid_loader)
    # NOTE: train_loader is intentionally NOT passed to accelerator.prepare.
    # Each rank owns a disjoint speaker-group shard of the manifest and must
    # consume ALL of its local batches; cross-rank batch sharding would drop
    # 7/8 of the data per epoch.

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

    feature_adapter_ref: list[S2MelFeatureAdapter | None] = [None]
    use_async_features = async_feature_extraction_enabled(cfg, accelerator)
    if accelerator.is_main_process and use_async_features:
        print("[Feature] Asynchronous extraction enabled (one batch ahead)")
    model.train()
    last_saved_step = global_step
    if _AUX_LOSS_TYPE:
        _init_aux_loss(
            cfg,
            accelerator.device,
            torch.bfloat16,
            world_size=accelerator.num_processes,
        )
    if _VOCODER_GAN is not None and resume_path is not None:
        _resume_vocoder_gan(resume_path, accelerator)
    # Per-rank optimizer steps per epoch (prepared loader is already sharded).
    steps_per_epoch = max(1, math.ceil(len(train_loader) / int(cfg.train.grad_accumulation)))
    if (
        resume_path is not None
        and resume_path.is_file()
        and args.resume_epoch_step is None
        and resume_meta is not None
        and resume_meta.get("epoch_step") is None
        and global_step > 0
    ):
        start_epoch = int(global_step) // int(steps_per_epoch)
        resume_epoch_step = int(global_step) % int(steps_per_epoch)
        if accelerator.is_main_process:
            print(
                f"[Resume] Inferred epoch={start_epoch + 1} epoch_step={resume_epoch_step} "
                f"from step={global_step} / steps_per_epoch={steps_per_epoch} "
                "(old .pth had no epoch_step)"
            )

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        if int(cfg.train.max_steps) > 0 and global_step >= int(cfg.train.max_steps):
            break
        epoch_step = 0
        epoch_loader = train_loader
        _set_loader_epoch(train_loader, epoch)
        if epoch == start_epoch and resume_epoch_step > 0:
            skip_batches = resume_epoch_step * int(cfg.train.grad_accumulation)
            epoch_loader = accelerator.skip_first_batches(train_loader, skip_batches)
            epoch_step = resume_epoch_step
            if accelerator.is_main_process:
                print(f"[Resume] Skipping first {skip_batches} batches of epoch {epoch + 1}")
        # Construct the DataLoader iterator before starting the feature thread.
        # On the first epoch this lets multiprocessing workers fork safely.
        raw_iterator = iter(epoch_loader)
        async_builder = None
        async_build_fn = lambda raw_batch: build_training_batch(
            raw_batch,
            cfg=cfg,
            accelerator=accelerator,
            feature_adapter_ref=feature_adapter_ref,
        )
        if use_async_features:
            async_builder = AsyncFeatureBatchBuilder(
                async_build_fn,
                device=accelerator.device,
            )
        async_skip_next = False

        try:
            current_raw_batch, exhausted = _next_raw_batch(raw_iterator)
            has_batch = _sync_has_batch(accelerator, not exhausted)
            if has_batch and async_builder is not None:
                if current_raw_batch is not None:
                    async_builder.submit(current_raw_batch)
                    current_raw_batch = None
                else:
                    async_skip_next = True

            while has_batch and not (
                int(cfg.train.max_steps) > 0
                and global_step >= int(cfg.train.max_steps)
            ):
                next_raw_batch = None
                has_next_batch = False
                prefetch_barrier = False
                validation_barrier = False
                with accelerator.accumulate(model):
                    if async_builder is not None:
                        if async_skip_next:
                            train_batch = None
                            async_skip_next = False
                        else:
                            try:
                                train_batch = async_builder.get(timeout=180.0)
                            except Exception as exc:
                                print(
                                    f"[DataLoader] feature get failed: "
                                    f"{type(exc).__name__}: {exc}",
                                    flush=True,
                                )
                                train_batch = None
                        next_global_step = global_step + int(accelerator.sync_gradients)
                        prefetch_barrier = step_requires_async_prefetch_barrier(
                            cfg,
                            sync_gradients=accelerator.sync_gradients,
                            next_global_step=next_global_step,
                            has_validation=valid_loader is not None,
                        )
                        valid_interval = int(cfg.train.valid_interval)
                        validation_barrier = (
                            accelerator.sync_gradients
                            and valid_loader is not None
                            and valid_interval > 0
                            and next_global_step % valid_interval == 0
                        )
                        if not prefetch_barrier:
                            next_raw_batch, next_exhausted = _next_raw_batch(raw_iterator)
                            if next_raw_batch is not None:
                                has_next_batch = True
                                async_builder.submit(next_raw_batch)
                            else:
                                has_next_batch = not next_exhausted
                                if has_next_batch:
                                    async_skip_next = True
                    else:
                        if current_raw_batch is None:
                            train_batch = None
                        else:
                            train_batch = _build_batch_or_none(async_build_fn, current_raw_batch)

                    _set_global_step(global_step)
                    if train_batch is None:
                        loss = torch.tensor(float("nan"), device=accelerator.device)
                    elif _AUX_LOSS_TYPE:
                        loss = forward_loss_with_aux(model, train_batch)
                    else:
                        loss = forward_loss(model, train_batch)
                    if train_batch is not None:
                        _spike_track_loss(loss, train_batch, global_step, accelerator)
                    # The skip decision MUST be identical on every rank: a rank
                    # that skips `backward` never joins the gradient all-reduce
                    # the other ranks are blocked in, which deadlocks NCCL until
                    # the 600 s watchdog kills the job.  Reduce a finiteness flag
                    # so all ranks skip together or none do.
                    _loss_finite = bool(torch.isfinite(loss))
                    if accelerator.num_processes > 1:
                        _flag = torch.tensor(
                            [1.0 if _loss_finite else 0.0], device=accelerator.device
                        )
                        torch.distributed.all_reduce(
                            _flag, op=torch.distributed.ReduceOp.MIN
                        )
                        _all_finite = bool(_flag.item() >= 1.0)
                        if _loss_finite and not _all_finite and accelerator.is_main_process:
                            print(
                                f"[SkipStep] step~{global_step} another rank saw a "
                                f"non-finite loss — all ranks skip together",
                                flush=True,
                            )
                    else:
                        _all_finite = _loss_finite
                    if not _all_finite:
                        # Skip BEFORE backward so no NaN gradients ever reach
                        # the GradScaler / optimizer; keep LR and step counters
                        # advancing.
                        if accelerator.is_main_process:
                            print(
                                f"[SkipStep] step~{global_step} loss non-finite "
                                f"— skipped before backward",
                                flush=True,
                            )
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        if _VOCODER_GAN is not None:
                            _VOCODER_GAN.zero_grad_all()
                    else:
                        accelerator.backward(loss)
                        if _VOCODER_GAN is not None:
                            # Own graph on detached audio, so it can run after the
                            # main backward has freed its own.
                            _VOCODER_GAN.discriminator_backward(
                                scale=1.0 / float(cfg.train.grad_accumulation)
                            )
                        _gn = None
                        _do_skip = False
                        if accelerator.sync_gradients and float(cfg.train.grad_clip) > 0:
                            _gn = accelerator.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
                            _spike_track_gnorm(_gn, global_step, accelerator)
                            _do_skip = _spike_should_skip_step(_gn)
                        if _do_skip:
                            # Grad explosion / NaN: skip the weight update so it
                            # cannot corrupt the model; keep LR advancing.
                            if accelerator.is_main_process:
                                _gnv = float(_gn) if _gn is not None else float("nan")
                                _ids = _spike_ids_from_batch(train_batch)
                                print(
                                    f"[SkipStep] step~{global_step} gnorm={_gnv:.3f} "
                                    f"exceeds SPIKE_SKIP_GNORM={_SPIKE_SKIP_GNORM} "
                                    f"— optimizer.step() skipped "
                                    f"n={len(_ids)} ids={_ids[:8]}",
                                    flush=True,
                                )
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            # The GradScaler's unscale_ was already consumed by
                            # clip_grad_norm_; take a no-op optimizer step so the
                            # scaler state advances without applying bad grads.
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            if _VOCODER_GAN is not None:
                                # The flow model's gradients exploded; do not feed
                                # the same step to the vocoder either.
                                _VOCODER_GAN.zero_grad_all()
                        else:
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            if _VOCODER_GAN is not None and accelerator.sync_gradients:
                                # Collective: every rank must reach this, since it
                                # all-reduces the vocoder and discriminator grads.
                                _VOCODER_GAN.clip_and_step()

                if validation_barrier and async_builder is not None:
                    # Validation workers are recreated on every pass. Tear down
                    # the producer before they fork and synchronously reuse the
                    # frozen adapter.
                    async_builder.close()
                    async_builder = None

                if accelerator.sync_gradients:
                    global_step += 1
                    epoch_step += 1
                    if global_step % int(cfg.train.log_interval) == 0:
                        reduced_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
                        lr = scheduler.get_last_lr()[0]
                        # Fractional completed epochs, e.g. 1.0 == first epoch done.
                        epoch_progress = epoch + min(1.0, epoch_step / steps_per_epoch)
                        if accelerator.is_main_process:
                            print(
                                f"[Train] epoch={epoch + 1} step={global_step} "
                                f"loss={reduced_loss:.5f} lr={lr:.3e}"
                            )
                        log_payload = {
                            "train/loss": reduced_loss,
                            "train/lr": lr,
                            "train/epoch": epoch_progress,
                        }
                        # Collective (gather) — every rank must reach this.
                        log_payload.update(_aux_metrics_for_log(accelerator))
                        accelerator.log(log_payload, step=global_step)

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

                if use_async_features:
                    if prefetch_barrier:
                        if int(cfg.train.max_steps) > 0 and global_step >= int(
                            cfg.train.max_steps
                        ):
                            has_next_batch = False
                        else:
                            next_raw_batch, next_exhausted = _next_raw_batch(raw_iterator)
                            if next_raw_batch is not None:
                                has_next_batch = True
                                if async_builder is None:
                                    async_builder = AsyncFeatureBatchBuilder(
                                        async_build_fn,
                                        device=accelerator.device,
                                    )
                                async_builder.submit(next_raw_batch)
                            else:
                                has_next_batch = not next_exhausted
                                if has_next_batch:
                                    async_skip_next = True
                    has_batch = _sync_has_batch(accelerator, has_next_batch)
                else:
                    current_raw_batch, exhausted = _next_raw_batch(raw_iterator)
                    has_batch = _sync_has_batch(accelerator, not exhausted)
        finally:
            if async_builder is not None:
                async_builder.close()

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



# --- Auxiliary loss support (env-var gated) -----------------------------------
import os as _aux_os

_AUX_LOSS_TYPE = _aux_os.environ.get("AUX_LOSS_TYPE", "bigvgan_mrstft")
_AUX_LOSS_WEIGHT = float(_aux_os.environ.get("AUX_LOSS_WEIGHT", "0.1"))
_FLOW_LOSS_WEIGHT = float(_aux_os.environ.get("FLOW_LOSS_WEIGHT", "1.0"))
_AUX_MRSTFT_WAVEL1 = float(_aux_os.environ.get("AUX_MRSTFT_WAVEL1", "0.0"))
# WaveFM keeps the phase term at 1.0.  Our phase comes from a frozen BigVGAN
# driven by the predicted mel, so it can be downweighted/disabled without
# touching the spectral terms (phase_l1 measured flat at the random-phase
# bound pi/2, see notes/tts-s2mel-quality.md).
_AUX_MRSTFT_PHASE_WEIGHT = float(_aux_os.environ.get("AUX_MRSTFT_PHASE_WEIGHT", "1.0"))
# Mel frames after the prompt that the frozen vocoder actually runs on.  128
# frames = 1.49 s at hop 512 / 44.1 kHz; raising it covers more of the target
# segment at a proportional cost in vocoder time and activation memory.
_AUX_MAX_CHUNK_FRAMES = int(_aux_os.environ.get("AUX_MAX_CHUNK_FRAMES", "128"))
# Draw the aux chunk start uniformly inside the target segment instead of always
# supervising its first _AUX_MAX_CHUNK_FRAMES frames.  Same cost, full coverage.
_AUX_CHUNK_RANDOM_OFFSET = _aux_os.environ.get("AUX_CHUNK_RANDOM_OFFSET", "0") == "1"
# Recompute the frozen vocoder's activations in the backward instead of storing
# them: 5.3x less aux-loss peak memory, bitwise-identical gradients.
_AUX_VOCODER_CKPT = _aux_os.environ.get("AUX_VOCODER_CKPT", "0") == "1"
_AUX_WAVLM_WEIGHT = float(_aux_os.environ.get("AUX_WAVLM_WEIGHT", "0.0"))
_AUX_WAVLM_MODEL = _aux_os.environ.get(
    "AUX_WAVLM_MODEL", "microsoft/wavlm-large"
)
_AUX_WAVLM_CACHE = _aux_os.environ.get("AUX_WAVLM_CACHE", "")
_AUX_WAVLM_LAYERS = tuple(
    int(x) for x in _aux_os.environ.get("AUX_WAVLM_LAYERS", "6,8,10,12").split(",")
    if x.strip()
)
_AUX_WAVLM_LOCAL = _aux_os.environ.get("AUX_WAVLM_LOCAL_ONLY", "0") == "1"
# The aux loss is computed on `x1_hat`, whose error is (1-t)*(v_pred - velocity).
# t is uniform, so at small t the one-step estimate is noise dominated and the
# vocoder-space target is unreachable -- an unlearnable fraction of every batch
# whose gradient still pushes the DiT.  Symptom in run as4v022n: over 32k steps
# neither term improved (flow valid 0.608 -> 0.658, aux mrstft 1.889 -> 1.938).
# Two independent knobs, usable together; both off by default.
#   AUX_T_MIN  drop samples with t <= AUX_T_MIN (also cheaper: the frozen
#              vocoder then runs on fewer samples)
#   AUX_T_POW  weight each surviving sample by t**AUX_T_POW (2 => t^2)
# Note AUX_T_POW only reaches the bigvgan_mrstft aux; the other aux types have
# no per-sample weighting and will raise if it is set.
# Default ON (kusuriuri, msg 935f815a): if the aux loss is on, the t gating goes
# with it, because at small t the vocoder-space target is unreachable.  Setting
# either variable to 0 explicitly still turns that half off.
_AUX_T_MIN = float(_aux_os.environ.get("AUX_T_MIN", "0.5"))
# t^p defaults on only for bigvgan_mrstft: it needs per-sample weighting, which
# no other aux type supports, and _init_aux_loss raises on that combination -- so
# a blanket default of 2 would make every other aux type fail at startup.  An
# explicit AUX_T_POW is still honoured, and still rejected where it cannot work.
_AUX_T_POW = float(
    _aux_os.environ.get(
        "AUX_T_POW", "2.0" if _AUX_LOSS_TYPE == "bigvgan_mrstft" else "0.0"
    )
)
_AUX_LOSS_MODULE = None
# Last step's flow/aux loss terms, refreshed by _record_aux_metrics.
_AUX_LAST_METRICS: dict = {}

# --- joint BigVGAN training (VOCODER_TRAIN=1, off by default) ------------------
# Unfreezing the vocoder is not a one-line change; see
# semantic2any/losses/vocoder_gan.py for why (moving MR-STFT target, DDP cannot
# wrap it, discriminators start from scratch).  Everything below is inert unless
# VOCODER_TRAIN=1, so a restart of a running job is unaffected.
_VOCODER_TRAIN = _aux_os.environ.get("VOCODER_TRAIN", "0") == "1"
_VOCODER_GAN = None
# The adversarial/feature-matching ramp needs the step count, which
# forward_loss_with_aux does not otherwise see.
_GLOBAL_STEP = 0


def _set_global_step(step: int) -> None:
    global _GLOBAL_STEP
    _GLOBAL_STEP = int(step)


def _init_aux_loss(cfg, device, dtype, world_size: int = 1):
    global _AUX_LOSS_MODULE
    if _AUX_LOSS_MODULE is not None:
        return
    if _AUX_T_POW != 0.0 and _AUX_LOSS_TYPE != "bigvgan_mrstft":
        # Fail at startup rather than with a TypeError 100 steps in: the other
        # aux types reduce over the batch with no per-sample axis to weight.
        raise RuntimeError(
            f"AUX_T_POW={_AUX_T_POW} needs per-sample weighting, which only "
            f"aux type 'bigvgan_mrstft' supports (got {_AUX_LOSS_TYPE!r}). "
            "AUX_T_MIN works with any type."
        )
    if _AUX_T_MIN > 0.0 or _AUX_T_POW != 0.0:
        print(
            f"[AuxLoss] t gating: AUX_T_MIN={_AUX_T_MIN} AUX_T_POW={_AUX_T_POW} "
            f"(aux supervises t>{_AUX_T_MIN}"
            + (f", weighted by t^{_AUX_T_POW}" if _AUX_T_POW else "")
            + ")",
            flush=True,
        )
    if _AUX_LOSS_TYPE == "mr_stft":
        from semantic2any.losses.auxiliary_losses import MultiResolutionMelLoss
        _AUX_LOSS_MODULE = MultiResolutionMelLoss(
            resolutions=(1, 2, 4, 8), sc_weight=1.0, mag_weight=1.0
        ).to(device)
        print(f"[AuxLoss] MultiResolutionMelLoss enabled, weight={_AUX_LOSS_WEIGHT}")
    elif _AUX_LOSS_TYPE == "bigvgan_loop":
        from semantic2any.losses.auxiliary_losses import BigVGANLoopLoss
        from semantic2any.third_party.indextts.bigvgan import BigVGAN
        vocoder_cfg = _get(cfg, "vocoder", None)
        model_id = (
            "nvidia/bigvgan_v2_44khz_128band_512x"
            if vocoder_cfg is None
            else str(_get(vocoder_cfg, "model_id", "") or "nvidia/bigvgan_v2_44khz_128band_512x")
        )
        cache_dir = str(_get(vocoder_cfg, "cache_dir", "") or "") if vocoder_cfg else ""
        load_kwargs = {}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        vocoder = BigVGAN.from_pretrained(model_id, **load_kwargs)
        vocoder = vocoder.to(device=device)  # keep float32 for stable vocoding
        vocoder.remove_weight_norm()
        vocoder.eval()
        preprocess = _get(cfg, "preprocess_params")
        sr = int(_get(preprocess, "sr", 44100))
        _AUX_LOSS_MODULE = BigVGANLoopLoss(
            vocoder=vocoder,
            sr=sr,
            n_fft_list=(2048, 1024, 512),
            hop_list=(512, 256, 128),
            win_list=(2048, 1024, 512),
        ).to(device)
        print(f"[AuxLoss] BigVGANLoopLoss enabled, weight={_AUX_LOSS_WEIGHT}")
    elif _AUX_LOSS_TYPE == "bigvgan_waveform":
        from semantic2any.losses.auxiliary_losses import BigVGANWaveformLoss
        from semantic2any.third_party.indextts.bigvgan import BigVGAN
        vocoder_cfg = _get(cfg, "vocoder", None)
        model_id = (
            "nvidia/bigvgan_v2_44khz_128band_512x"
            if vocoder_cfg is None
            else str(_get(vocoder_cfg, "model_id", "") or "nvidia/bigvgan_v2_44khz_128band_512x")
        )
        cache_dir = str(_get(vocoder_cfg, "cache_dir", "") or "") if vocoder_cfg else ""
        load_kwargs = {}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        vocoder = BigVGAN.from_pretrained(model_id, **load_kwargs)
        vocoder = vocoder.to(device=device)
        vocoder.remove_weight_norm()
        vocoder.eval()
        _AUX_LOSS_MODULE = BigVGANWaveformLoss(
            vocoder=vocoder,
            sr=int(_get(_get(cfg, "preprocess_params"), "sr", 44100)),
        ).to(device)
        print(f"[AuxLoss] BigVGANWaveformLoss enabled, weight={_AUX_LOSS_WEIGHT}")
    elif _AUX_LOSS_TYPE == "bigvgan_mrstft":
        from semantic2any.losses.auxiliary_losses import BigVGANMRSTFTLoss
        from semantic2any.third_party.indextts.bigvgan import BigVGAN
        vocoder_cfg = _get(cfg, "vocoder", None)
        model_id = (
            "nvidia/bigvgan_v2_44khz_128band_512x"
            if vocoder_cfg is None
            else str(_get(vocoder_cfg, "model_id", "") or "nvidia/bigvgan_v2_44khz_128band_512x")
        )
        cache_dir = str(_get(vocoder_cfg, "cache_dir", "") or "") if vocoder_cfg else ""
        load_kwargs = {}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        vocoder = BigVGAN.from_pretrained(model_id, **load_kwargs)
        vocoder = vocoder.to(device=device)
        spect = _get(_get(cfg, "preprocess_params"), "spect_params")
        hop_size = int(_get(spect, "hop_length", 512))
        if _VOCODER_TRAIN:
            # BigVGAN is trained *with* weight norm; folding it away first would
            # train the folded weights and make the result un-resumable, so
            # remove_weight_norm() is deliberately skipped here.  (.eval() is
            # skipped for symmetry only -- this net has no BatchNorm/Dropout.)
            vocoder.train()
        else:
            vocoder.remove_weight_norm()
            vocoder.eval()
        _AUX_LOSS_MODULE = BigVGANMRSTFTLoss(
            vocoder=vocoder,
            sr=int(_get(_get(cfg, "preprocess_params"), "sr", 44100)),
            wave_l1_weight=_AUX_MRSTFT_WAVEL1,
            stft_kwargs={"phase_weight": _AUX_MRSTFT_PHASE_WEIGHT},
            max_chunk_frames=_AUX_MAX_CHUNK_FRAMES,
            random_chunk_offset=_AUX_CHUNK_RANDOM_OFFSET,
            checkpoint_vocoder=_AUX_VOCODER_CKPT,
            trainable_vocoder=_VOCODER_TRAIN,
            vocode_real_mel=_VOCODER_TRAIN,
            hop_size=hop_size,
            wavlm_weight=_AUX_WAVLM_WEIGHT,
            wavlm_model_id=_AUX_WAVLM_MODEL,
            wavlm_cache_dir=_AUX_WAVLM_CACHE,
            wavlm_layers=_AUX_WAVLM_LAYERS,
            wavlm_local_files_only=_AUX_WAVLM_LOCAL,
        ).to(device)
        print(
            f"[AuxLoss] BigVGANMRSTFTLoss (WaveFM) enabled, weight={_AUX_LOSS_WEIGHT}, "
            f"wave_l1_weight={_AUX_MRSTFT_WAVEL1}, "
            f"phase_weight={_AUX_MRSTFT_PHASE_WEIGHT}, "
            f"max_chunk_frames={_AUX_MAX_CHUNK_FRAMES}, "
            f"random_chunk_offset={_AUX_CHUNK_RANDOM_OFFSET}, "
            f"checkpoint_vocoder={_AUX_VOCODER_CKPT}, "
            f"wavlm_weight={_AUX_WAVLM_WEIGHT} layers={_AUX_WAVLM_LAYERS}"
        )
    if _VOCODER_TRAIN:
        _init_vocoder_gan(cfg, device, world_size)


def _mel_args_from_cfg(cfg) -> dict:
    """The mel contract, read from the same config the dataset reads.

    Must stay identical to semantic2any/data/s2mel_dataset.py's ``mel_args`` --
    a mel loss computed with a different convention than the training mel would
    optimise the vocoder against a target the model can never produce.
    """
    preprocess = _get(cfg, "preprocess_params")
    spect = _get(preprocess, "spect_params")
    # `fmax: None` in the yaml is the *string* "None" -- yaml only spells null as
    # `null` or `~`.  Same coercion as build_dataset above; getting it wrong
    # turns fmax into a float and silently changes the filterbank.
    fmax = _get(spect, "fmax", "None")
    return {
        "n_fft": int(_get(spect, "n_fft", 2048)),
        "num_mels": int(_get(spect, "n_mels", 128)),
        "sampling_rate": int(_get(preprocess, "sr", 44100)),
        "hop_size": int(_get(spect, "hop_length", 512)),
        "win_size": int(_get(spect, "win_length", 2048)),
        "fmin": float(_get(spect, "fmin", 0.0)),
        "fmax": None if fmax in (None, "None", "", "null") else float(fmax),
        "center": False,
    }


def _resume_vocoder_gan(resume_path: Path, accelerator) -> None:
    """Restore the trained vocoder + discriminators, or refuse to guess.

    Resuming a joint run against the pristine HF vocoder would silently undo
    every vocoder step taken so far, so a missing file is an error rather than a
    warning -- except when starting joint training from a frozen-vocoder run,
    which is the one legitimate case and needs VOCODER_RESUME_FRESH_GAN=1.
    """
    # Only the accelerator checkpoint directory carries joint state; a
    # weights-only s2mel_step*.pth never does.
    state_path = resume_path / "vocoder_gan.pt"
    if not state_path.is_file():
        if _aux_os.environ.get("VOCODER_RESUME_FRESH_GAN", "0") == "1":
            if accelerator.is_main_process:
                print(
                    f"[Vocoder] {state_path.name} absent; starting joint training "
                    "from the pretrained HF vocoder with fresh discriminators",
                    flush=True,
                )
            return
        raise FileNotFoundError(
            f"VOCODER_TRAIN=1 but {state_path} does not exist. Resuming here "
            "would load the untrained HF vocoder and throw away every vocoder "
            "step in this run. Set VOCODER_RESUME_FRESH_GAN=1 only if this "
            "checkpoint really predates joint training."
        )
    state = torch.load(state_path, map_location=accelerator.device)
    _VOCODER_GAN.load_training_state(state)
    if accelerator.is_main_process:
        print(f"[Vocoder] restored joint state from {state_path}", flush=True)


def _init_vocoder_gan(cfg, device, world_size: int) -> None:
    """Build the joint-training state around the aux loss's vocoder."""
    global _VOCODER_GAN
    if _VOCODER_GAN is not None:
        return
    if _AUX_LOSS_TYPE != "bigvgan_mrstft":
        raise RuntimeError(
            f"VOCODER_TRAIN=1 needs AUX_LOSS_TYPE=bigvgan_mrstft (got "
            f"{_AUX_LOSS_TYPE!r}); the other aux types do not expose the "
            "waveform chunks the vocoder/GAN losses need"
        )
    if str(_get(cfg.train, "mixed_precision", "no")).lower() == "fp16":
        # fp16 puts a GradScaler between the loss and the grads; the
        # discriminator's backward is ours, outside accelerate, so its gradients
        # would be unscaled while the vocoder's are scaled.  bf16 needs no
        # scaler and sidesteps this entirely.
        raise RuntimeError(
            "VOCODER_TRAIN=1 requires train.mixed_precision=bf16 (or no); with "
            "fp16 the discriminator's own backward bypasses accelerate's "
            "GradScaler and the two halves of the GAN see different gradient "
            "scales"
        )
    from semantic2any.losses.vocoder_gan import VocoderGANTrainer

    _VOCODER_GAN = VocoderGANTrainer.from_env(
        _AUX_LOSS_MODULE.vocoder,
        mel_args=_mel_args_from_cfg(cfg),
        world_size=int(world_size),
    ).to(device)
    _VOCODER_GAN.sync_initial_weights()
    print(_VOCODER_GAN.describe(), flush=True)


def forward_loss_with_aux(model, batch):
    """forward_loss with optional auxiliary loss."""
    loss, x1_hat = model(
        batch["mel"],
        batch["mel_lens"],
        batch["prompt_lens"],
        batch["semantic"],
        batch["style"],
        semantic_is_mu=False,
        semantic_lens=batch.get("semantic_lens"),
        prompt_semantic_lens=batch.get("prompt_semantic_lens"),
    )
    if _AUX_LOSS_MODULE is not None and x1_hat is not None:
        selection = _aux_t_selection(model, x1_hat.size(0), x1_hat.device)
        if selection is None:
            # Every sample in this batch drew t <= AUX_T_MIN, so there is no
            # trustworthy aux signal.  Report aux as zero and keep the flow
            # term: the step still trains, and skipping it would make the
            # skip decision rank-local (see _sync_has_batch).
            aux_loss = x1_hat.new_zeros(())
            if hasattr(_AUX_LOSS_MODULE, "zero_components"):
                _AUX_LOSS_MODULE.zero_components(x1_hat.device, x1_hat.dtype)
        else:
            keep, weights = selection
            extra = {} if weights is None else {"sample_weights": weights}
            if _VOCODER_TRAIN:
                target_wav = batch.get("target_wav")
                if target_wav is None:
                    raise RuntimeError(
                        "VOCODER_TRAIN=1 but the batch has no 'target_wav'. Both "
                        "paired extraction paths must attach it: the main-process "
                        "one in S2MelFeatureAdapter.extract_paired_from_audio_paths "
                        "(used when data.extract_mel_in_worker is false, which is "
                        "the default) and the worker one in "
                        "S2MelPairedDataset._attach_paired_worker_features. Both "
                        "gate on VOCODER_TRAIN, which must therefore reach the "
                        "dataloader workers too."
                    )
                extra["target_wav"] = target_wav[keep]
                wav_lens = batch.get("target_wav_lens")
                if wav_lens is not None:
                    extra["target_wav_lens"] = wav_lens[keep]
            aux_loss = _AUX_LOSS_MODULE(
                x1_hat[keep],
                batch["mel"][keep],
                batch["mel_lens"][keep],
                batch["prompt_lens"][keep],
                **extra,
            )
        vocoder_loss = _vocoder_gan_loss()
        _record_aux_metrics(loss, aux_loss, vocoder_loss)
        loss = _FLOW_LOSS_WEIGHT * loss + _AUX_LOSS_WEIGHT * aux_loss
        if vocoder_loss is not None:
            loss = loss + vocoder_loss
    return loss


def _vocoder_gan_loss():
    """Vocoder-side (and discriminator-side) losses for this micro-batch.

    Returns None whenever there is nothing to train on -- joint training off, or
    the t gate dropped every sample so the aux loss produced no audio.  Note the
    vocoder terms are *not* scaled by AUX_LOSS_WEIGHT: they carry BigVGAN's own
    weights (mel 15, fm 2, adv 1) against real audio, independent of how much the
    flow model is allowed to hear from the aux term.
    """
    if _VOCODER_GAN is None or _AUX_LOSS_MODULE is None:
        return None
    waveforms = getattr(_AUX_LOSS_MODULE, "last_waveforms", {})
    if not waveforms or "real_wav" not in waveforms:
        return None
    return _VOCODER_GAN.generator_loss(
        wav_real=waveforms["real_wav"],
        wav_from_real_mel=waveforms.get("real_mel_wav"),
        wav_from_pred_mel=waveforms.get("pred_wav"),
        real_mel_chunk=waveforms.get("real_mel_chunk"),
        global_step=_GLOBAL_STEP,
    )


def _cfm_module(model):
    """Reach the CFM through the accelerate / DDP wrappers."""
    inner = model
    for _ in range(4):
        nxt = getattr(inner, "module", None)
        if nxt is None:
            break
        inner = nxt
    # model.models is an nn.ModuleDict, which supports `in` and [] but not .get.
    models = getattr(inner, "models", None)
    cfm = models["cfm"] if models is not None and "cfm" in models else None
    if cfm is None:
        raise RuntimeError(
            "AUX_T_MIN/AUX_T_POW are set but the CFM module could not be reached "
            f"through {type(model).__name__} to read its sampled t"
        )
    return cfm


def _aux_t_selection(model, batch_size: int, device):
    """Which samples the aux term supervises, and their per-sample weights.

    Returns (index, weights) or None when nothing survives the AUX_T_MIN gate.
    `index` is `slice(None)` when the gate is off, so there is one code path.
    """
    if _AUX_T_MIN <= 0.0 and _AUX_T_POW == 0.0:
        return (slice(None), None)
    t = getattr(_cfm_module(model), "last_time", None)
    if t is None:
        raise RuntimeError(
            "AUX_T_MIN/AUX_T_POW are set but the CFM did not stash `last_time`; "
            "the flow_matching.py half of the t-gating patch is missing"
        )
    t = t.detach().reshape(-1).to(device=device, dtype=torch.float32)
    if t.numel() != batch_size:
        raise RuntimeError(
            f"CFM stashed {t.numel()} timesteps but the batch has {batch_size}; "
            "last_time is stale, so the aux weighting would be misaligned"
        )
    if _AUX_T_MIN > 0.0:
        keep = t > _AUX_T_MIN
        if not bool(keep.any()):
            return None
    else:
        keep = slice(None)
    weights = None if _AUX_T_POW == 0.0 else t[keep].clamp_min(0.0) ** _AUX_T_POW
    return (keep, weights)


def _record_aux_metrics(flow_loss, aux_loss, vocoder_loss=None):
    """Stash this step's flow/aux loss terms for the next wandb log."""
    metrics = {"flow": flow_loss.detach(), "aux": aux_loss.detach()}
    for name, value in getattr(_AUX_LOSS_MODULE, "last_components", {}).items():
        metrics[f"aux/{name}"] = value
    if _VOCODER_GAN is not None:
        zero = flow_loss.detach().new_zeros(())
        metrics["vocoder/total"] = (
            zero if vocoder_loss is None else vocoder_loss.detach()
        )
        for name, value in _VOCODER_GAN.last_components.items():
            metrics[f"vocoder/{name}"] = value
    _AUX_LAST_METRICS.clear()
    _AUX_LAST_METRICS.update(metrics)


def _aux_log_keys():
    """Stable key order, identical on every rank (needed for the gather)."""
    keys = ["flow", "aux"]
    keys.extend(
        f"aux/{name}"
        for name in getattr(_AUX_LOSS_MODULE, "component_keys", ())
    )
    if _VOCODER_GAN is not None:
        keys.append("vocoder/total")
        keys.extend(f"vocoder/{name}" for name in _VOCODER_GAN.component_keys)
    return keys


def _aux_metrics_for_log(accelerator):
    """Cross-rank mean of every aux loss component, as wandb metrics.

    Collective: must be called by all ranks at the same step.
    """
    if _AUX_LOSS_MODULE is None or not _AUX_LAST_METRICS:
        return {}
    keys = _aux_log_keys()
    zero = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    values = torch.stack(
        [_AUX_LAST_METRICS.get(key, zero).detach().float().reshape(()) for key in keys]
    )
    gathered = accelerator.gather(values.unsqueeze(0)).reshape(-1, len(keys)).mean(dim=0)
    return {f"train/{key}": float(value) for key, value in zip(keys, gathered)}
# --- end auxiliary loss support ------------------------------------------------
if __name__ == "__main__":
    main()
