"""Precompute s2mel training features (mel / semantic / style) offline.

Extracting w2v-bert + semantic-codec + campplus features online inside the
training loop re-does the same GPU work every epoch and serializes training.
This script materializes the features once and writes a JSONL manifest that
`S2MelJsonlDataset` + `S2MelCollator` consume directly (precomputed path).

Example:
    python scripts/precompute_s2mel_features.py \
        --config configs/s2mel_zipformer_vctk10pct_20260709_e30_bs32_8gpu.yaml \
        --source datasets/vctk_10pct_speaker_stratified/metadata/vctk_train_minus_test_10pct.jsonl \
        --output-dir datasets/vctk_10pct_features/train \
        --device cuda:0 --batch-size 16 --num-workers 8

    # Optional manual parallelism over GPUs:
    #   --num-shards 4 --shard 0  (repeat with shard 1..3 on other GPUs)
    # then: cat manifest.shard*of4.jsonl > manifest.jsonl

Train afterwards with:
    --train-jsonl <output-dir>/manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from semantic2any.data.s2mel_dataset import S2MelJsonlDataset, S2MelSpeechDataDataset
from semantic2any.utils.indextts_adapters import build_feature_adapter
from semantic2any.utils.semantic_codecs import (
    prepare_feature_metadata,
    resolve_semantic_codec_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute mel/semantic/style features for s2mel training.")
    parser.add_argument("--config", default="configs/s2mel_zipformer.yaml")
    parser.add_argument("--source", required=True, help="SpeechData dir/metadata JSONL or audio manifest JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--semantic-codec", choices=("maskgct", "sac"), default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _looks_like_speechdata_source(source: str) -> bool:
    if source.startswith("gs://"):
        return True
    path = Path(source).expanduser()
    if path.is_file():
        return path.parent.name == "metadata"
    if path.is_dir():
        return (path / "metadata").is_dir() or path.name == "metadata"
    return False


def _identity_collate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return records


def _record_id(record: dict[str, Any]) -> str:
    raw = record.get("id") or record.get("audio_path") or f"line{record.get('_line_no', 0)}"
    return str(raw).replace("/", "_").replace(":", "_")


def _feature_paths(feats_dir: Path, utt_id: str) -> dict[str, Path]:
    bucket = hashlib.sha1(utt_id.encode("utf-8")).hexdigest()[:2]
    base = feats_dir / bucket
    return {
        "mel_path": base / f"{utt_id}.mel.pt",
        "semantic_path": base / f"{utt_id}.semantic.pt",
        "style_path": base / f"{utt_id}.style.pt",
    }


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(tensor, tmp)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("--shard must be in [0, --num-shards)")

    cfg = OmegaConf.load(args.config)
    resolve_semantic_codec_config(cfg, args.semantic_codec)
    if args.source and _looks_like_speechdata_source(args.source):
        dataset: Any = S2MelSpeechDataDataset(
            args.source, cache_dir=str(cfg.data.speechdata_cache_dir or "") or None
        )
    else:
        dataset = S2MelJsonlDataset(args.source)

    indices = list(range(args.shard, len(dataset), args.num_shards))
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_identity_collate,
    )

    output_dir = Path(args.output_dir).expanduser()
    feats_dir = output_dir / "feats"
    output_dir.mkdir(parents=True, exist_ok=True)
    codec_metadata = prepare_feature_metadata(
        output_dir, cfg, overwrite=args.overwrite
    )

    device = torch.device(args.device)
    adapter = build_feature_adapter(cfg).to(device)
    adapter.eval()
    manifest_name = args.manifest_name
    if args.num_shards > 1:
        stem, suffix = Path(manifest_name).stem, Path(manifest_name).suffix or ".jsonl"
        manifest_name = f"{stem}.shard{args.shard}of{args.num_shards}{suffix}"
    manifest_path = output_dir / manifest_name

    written = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for records in tqdm(loader, desc=f"precompute shard {args.shard}/{args.num_shards}"):
            todo: list[tuple[dict[str, Any], dict[str, Path]]] = []
            for record in records:
                paths = _feature_paths(feats_dir, _record_id(record))
                if not args.overwrite and all(p.is_file() and p.stat().st_size > 0 for p in paths.values()):
                    skipped += 1
                else:
                    todo.append((record, paths))
                entry = {
                    "id": record.get("id"),
                    "audio_path": record.get("_speechdata_audio_path") or record.get("audio_path"),
                    "mel_path": str(paths["mel_path"].relative_to(output_dir)),
                    "semantic_path": str(paths["semantic_path"].relative_to(output_dir)),
                    "style_path": str(paths["style_path"].relative_to(output_dir)),
                    "semantic_codec": codec_metadata["name"],
                    "semantic_dim": codec_metadata["semantic_dim"],
                    "semantic_fps": codec_metadata["semantic_fps"],
                    "semantic_fingerprint": codec_metadata["fingerprint"],
                }
                for key in ("text", "speaker_id", "duration", "language"):
                    if key in record:
                        entry[key] = record[key]
                manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if not todo:
                continue
            features = adapter.extract_utterance_features([r["audio_path"] for r, _ in todo])
            for (_, paths), item in zip(todo, features, strict=True):
                _save_tensor(paths["mel_path"], item["mel"].half().cpu())
                _save_tensor(paths["semantic_path"], item["semantic"].half().cpu())
                _save_tensor(paths["style_path"], item["style"].float().cpu())
                written += 1

    print(f"[Done] extracted={written} reused={skipped} manifest={manifest_path}")


if __name__ == "__main__":
    main()
