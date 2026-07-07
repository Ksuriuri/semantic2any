from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from semantic2any.data.s2mel_dataset import S2MelCollator, S2MelJsonlDataset
from semantic2any.models import Semantic2MelModel
from semantic2any.utils.checkpoint import load_compatible_checkpoint, save_compatible_checkpoint
from semantic2any.utils.indextts_adapters import IndexTTSFeatureAdapter, move_feature_batch_to_device


def _get(obj, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IndexTTS2.5-style ZipFormer semantic2mel.")
    parser.add_argument("--config", default="configs/s2mel_zipformer.yaml")
    parser.add_argument("--train-jsonl", default=None)
    parser.add_argument("--valid-jsonl", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--indextts-root", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg, args: argparse.Namespace):
    if args.train_jsonl is not None:
        cfg.data.train_jsonl = args.train_jsonl
    if args.valid_jsonl is not None:
        cfg.data.valid_jsonl = args.valid_jsonl
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
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloader(cfg, manifest: str, shuffle: bool) -> DataLoader:
    dataset = S2MelJsonlDataset(manifest)
    spect = cfg.preprocess_params.spect_params
    collator = S2MelCollator(
        hop_length=int(spect.hop_length),
        sample_rate=int(cfg.preprocess_params.sr),
        min_prompt_seconds=float(cfg.data.min_prompt_seconds),
        max_prompt_seconds=float(cfg.data.max_prompt_seconds),
        min_generated_frames=int(cfg.data.min_generated_frames),
    )
    kwargs: dict[str, Any] = {}
    if int(cfg.data.num_workers) > 0:
        kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
        kwargs["persistent_workers"] = True
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


def rotate_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    stale = ckpts[: max(0, len(ckpts) - keep_last)]
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
    weights = sorted(output_dir.glob("s2mel_step*.pth"), key=lambda p: p.stat().st_mtime)
    for path in weights[: max(0, len(weights) - keep_last)]:
        path.unlink(missing_ok=True)


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
    return feature_adapter_ref[0].extract_from_audio_paths(batch["audio_paths"])


def forward_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    loss, _ = model(
        batch["mel"],
        batch["mel_lens"],
        batch["prompt_lens"],
        batch["semantic"],
        batch["style"],
        semantic_is_mu=False,
    )
    return loss


@torch.no_grad()
def validate(model, loader, cfg, accelerator: Accelerator, feature_adapter_ref) -> float:
    model.eval()
    losses = []
    for batch in loader:
        train_batch = build_training_batch(
            batch,
            cfg=cfg,
            accelerator=accelerator,
            feature_adapter_ref=feature_adapter_ref,
        )
        loss = forward_loss(model, train_batch)
        losses.append(accelerator.gather_for_metrics(loss.detach()).mean())
    model.train()
    if not losses:
        return float("nan")
    return torch.stack(losses).mean().item()


def save_training_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    cfg,
    output_dir: Path,
    epoch: int,
    global_step: int,
) -> None:
    save_dir = output_dir / f"checkpoint-{global_step}"
    accelerator.save_state(str(save_dir))
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_compatible_checkpoint(
            output_dir / f"s2mel_step{global_step}.pth",
            unwrapped,
            epoch=epoch,
            step=global_step,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        rotate_checkpoints(output_dir, int(cfg.train.keep_last))
        print(f"[Checkpoint] Saved {save_dir}")


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(OmegaConf.load(args.config), args)
    if not cfg.data.train_jsonl:
        raise ValueError("Set data.train_jsonl or pass --train-jsonl")

    set_seed(int(cfg.seed))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.train.grad_accumulation),
        mixed_precision=str(cfg.train.mixed_precision),
        log_with=None if bool(cfg.train.no_wandb) else "wandb",
    )
    if not bool(cfg.train.no_wandb):
        accelerator.init_trackers(
            "semantic2any-s2mel",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    output_dir = Path(cfg.train.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.resolved.yaml")
    accelerator.wait_for_everyone()

    train_loader = make_dataloader(cfg, cfg.data.train_jsonl, shuffle=True)
    valid_loader = make_dataloader(cfg, cfg.data.valid_jsonl, shuffle=False) if cfg.data.valid_jsonl else None

    model = Semantic2MelModel(cfg.s2mel)
    resume_from = str(cfg.train.resume_from or "")
    start_epoch = 0
    global_step = 0
    if resume_from and Path(resume_from).is_file():
        start_epoch, global_step = load_compatible_checkpoint(model, resume_from, strict=False)
        if accelerator.is_main_process:
            print(f"[Resume] Loaded compatible checkpoint {resume_from} at step={global_step}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.weight_decay),
    )
    updates_per_epoch = math.ceil(len(train_loader) / int(cfg.train.grad_accumulation))
    total_steps = int(cfg.train.max_steps) if int(cfg.train.max_steps) > 0 else int(cfg.train.epochs) * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.train.warmup_steps),
        num_training_steps=max(1, total_steps),
    )

    if valid_loader is None:
        model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    else:
        model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, valid_loader, scheduler
        )

    if resume_from and Path(resume_from).is_dir():
        accelerator.load_state(resume_from)
        if accelerator.is_main_process:
            print(f"[Resume] Loaded accelerator state {resume_from}")

    feature_adapter_ref: list[IndexTTSFeatureAdapter | None] = [None]
    model.train()
    last_saved_step = global_step

    for epoch in range(start_epoch, int(cfg.train.epochs)):
        for raw_batch in train_loader:
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
                reduced_loss = accelerator.gather_for_metrics(loss.detach()).mean().item()
                if global_step % int(cfg.train.log_interval) == 0:
                    lr = scheduler.get_last_lr()[0]
                    if accelerator.is_main_process:
                        print(f"[Train] epoch={epoch + 1} step={global_step} loss={reduced_loss:.5f} lr={lr:.3e}")
                    accelerator.log({"train/loss": reduced_loss, "train/lr": lr}, step=global_step)

                if valid_loader is not None and global_step % int(cfg.train.valid_interval) == 0:
                    val_loss = validate(model, valid_loader, cfg, accelerator, feature_adapter_ref)
                    if accelerator.is_main_process:
                        print(f"[Valid] step={global_step} loss={val_loss:.5f}")
                    accelerator.log({"valid/loss": val_loss}, step=global_step)

                if global_step % int(cfg.train.save_interval) == 0:
                    save_training_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        cfg=cfg,
                        output_dir=output_dir,
                        epoch=epoch,
                        global_step=global_step,
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
