#!/usr/bin/env python3
"""Precompute compact MaskGCT semantic indices into resumable binary shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from semantic2any.data.s2mel_dataset import S2MelJsonlDataset
from semantic2any.utils.semantic_codecs import (
    MaskGCTSemanticCodec,
    build_semantic_codec,
    resolve_semantic_codec_config,
    semantic_codec_info,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one uint16 MaskGCT code per 50 Hz semantic frame."
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Input JSONL or directory. Repeat to combine sources into one output manifest.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/s2mel_zipformer.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--max-audio-seconds", type=float, default=None)
    parser.add_argument("--skip-audio-errors", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class RecordDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.records[index])


class AudioCollator:
    def __init__(self, max_audio_seconds: float, skip_errors: bool) -> None:
        self.max_audio_seconds = max_audio_seconds
        self.skip_errors = skip_errors

    def __call__(
        self, records: list[dict[str, Any]]
    ) -> tuple[
        list[dict[str, Any]],
        list[torch.Tensor],
        list[int],
        list[dict[str, Any]],
    ]:
        valid_records: list[dict[str, Any]] = []
        waveforms: list[torch.Tensor] = []
        sample_rates: list[int] = []
        failures: list[dict[str, Any]] = []
        for record in records:
            path = str(record["audio_path"])
            try:
                audio, sample_rate = torchaudio.load(path)
                if audio.size(0) > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                max_samples = int(self.max_audio_seconds * sample_rate)
                audio = audio[:, :max_samples].float()
                if audio.numel() == 0:
                    raise ValueError("decoded audio is empty")
            except (OSError, RuntimeError, ValueError) as exc:
                if not self.skip_errors:
                    raise
                failures.append(
                    {
                        "source_index": int(record["_source_index"]),
                        "source_identity": _source_identity(record),
                        "audio_path": path,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                warnings.warn(f"Skipping undecodable audio {path}: {exc}", RuntimeWarning)
                continue
            valid_records.append(record)
            waveforms.append(audio.contiguous())
            sample_rates.append(int(sample_rate))
        return valid_records, waveforms, sample_rates, failures


class GPUWaveformResampler:
    """Batch worker-decoded variable-length audio and resample on the target GPU."""

    def __init__(self, device: torch.device, target_rate: int = 16000) -> None:
        self.device = torch.device(device)
        self.target_rate = int(target_rate)
        self._cache: dict[tuple[int, torch.dtype], torchaudio.transforms.Resample] = {}

    def _resampler(
        self,
        source_rate: int,
        dtype: torch.dtype,
    ) -> torchaudio.transforms.Resample:
        key = (source_rate, dtype)
        resampler = self._cache.get(key)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(
                source_rate,
                self.target_rate,
                dtype=dtype,
            ).to(self.device)
            resampler.eval()
            self._cache[key] = resampler
        return resampler

    @torch.no_grad()
    def __call__(
        self,
        waveforms: list[torch.Tensor],
        sample_rates: list[int],
    ) -> list[np.ndarray]:
        if not waveforms or len(waveforms) != len(sample_rates):
            raise ValueError("waveforms and sample_rates must have the same non-zero length")
        grouped: dict[int, list[int]] = {}
        for index, sample_rate in enumerate(sample_rates):
            grouped.setdefault(int(sample_rate), []).append(index)

        outputs: list[np.ndarray | None] = [None] * len(waveforms)
        for source_rate, indices in grouped.items():
            source_items = [waveforms[index].squeeze(0).float() for index in indices]
            source_lengths = [item.numel() for item in source_items]
            padded = pad_sequence(
                source_items,
                batch_first=True,
                padding_value=0.0,
            ).to(self.device, non_blocking=True)
            resampled = (
                padded
                if source_rate == self.target_rate
                else self._resampler(source_rate, padded.dtype)(padded)
            )
            for local_index, (global_index, source_length) in enumerate(
                zip(indices, source_lengths, strict=True)
            ):
                target_length = math.ceil(
                    source_length * self.target_rate / source_rate
                )
                outputs[global_index] = (
                    resampled[local_index, :target_length]
                    .detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
        if any(item is None for item in outputs):
            raise RuntimeError("GPU resampling did not produce every waveform")
        return [item for item in outputs if item is not None]


def _source_identity(record: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "id": record.get("id"),
            "audio_path": record.get("audio_path"),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    tmp_path.replace(path)


def _install_lookup_once(path: Path, lookup: torch.Tensor) -> None:
    if path.is_file():
        return
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save({"lookup": lookup.float().cpu().contiguous()}, tmp_path)
    try:
        os.link(tmp_path, path)
    except FileExistsError:
        pass
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_jsonl(
    path: Path,
    *,
    repair_partial_tail: bool = False,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    mode = "r+b" if repair_partial_tail else "rb"
    with path.open(mode) as file_obj:
        line_number = 0
        while True:
            line_start = file_obj.tell()
            raw_line = file_obj.readline()
            if not raw_line:
                break
            line_number += 1
            try:
                line = raw_line.decode("utf-8").strip()
            except UnicodeDecodeError:
                is_final_line = file_obj.tell() == path.stat().st_size
                if not repair_partial_tail or not is_final_line:
                    raise
                file_obj.truncate(line_start)
                file_obj.flush()
                os.fsync(file_obj.fileno())
                break
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                is_final_line = file_obj.tell() == path.stat().st_size
                if not repair_partial_tail or not is_final_line:
                    raise
                file_obj.truncate(line_start)
                file_obj.flush()
                os.fsync(file_obj.fileno())
                break
            row["_journal_line"] = line_number
            rows.append(row)
    return rows


def _append_jsonl(file_obj, row: dict[str, Any]) -> None:
    file_obj.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    file_obj.flush()
    os.fsync(file_obj.fileno())


def _load_records(sources: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        dataset = S2MelJsonlDataset(source)
        records.extend(dataset[index] for index in range(len(dataset)))
    for index, record in enumerate(records):
        record["_source_index"] = index
    return records


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise ValueError("--shard must be in [0, --num-shards)")
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative")

    cfg = OmegaConf.load(args.config)
    if args.model_dir is not None:
        cfg.paths.model_dir = args.model_dir
    info = resolve_semantic_codec_config(cfg, "maskgct")
    max_audio_seconds = (
        float(args.max_audio_seconds)
        if args.max_audio_seconds is not None
        else float(cfg.data.max_audio_seconds)
    )
    if max_audio_seconds <= 0:
        raise ValueError("--max-audio-seconds must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    codes_dir = output_dir / "codes"
    manifests_dir = output_dir / "manifests"
    errors_dir = output_dir / "errors"
    for directory in (codes_dir, manifests_dir, errors_dir):
        directory.mkdir(parents=True, exist_ok=True)
    suffix = f"shard{args.shard:05d}of{args.num_shards:05d}"
    binary_path = codes_dir / f"codes.{suffix}.bin"
    manifest_path = manifests_dir / f"manifest.{suffix}.jsonl"
    errors_path = errors_dir / f"errors.{suffix}.jsonl"
    lookup_path = output_dir / "maskgct_lookup.pt"
    metadata_path = output_dir / "semantic_code_metadata.json"

    if args.overwrite:
        for path in (binary_path, manifest_path, errors_path):
            path.unlink(missing_ok=True)

    records = _load_records(args.source)
    existing_rows = _load_jsonl(manifest_path, repair_partial_tail=True)
    error_rows = _load_jsonl(errors_path, repair_partial_tail=True)
    done: dict[int, str] = {}
    committed_tokens = 0
    for row in existing_rows:
        source_index = int(row["source_index"])
        identity = str(row["source_identity"])
        if source_index in done:
            raise ValueError(f"Duplicate source_index {source_index} in {manifest_path}")
        done[source_index] = identity
        offset = int(row["semantic_code_offset"])
        length = int(row["semantic_code_length"])
        if offset != committed_tokens or length <= 0:
            raise ValueError(f"Non-contiguous or empty code entry in {manifest_path}")
        committed_tokens = offset + length
    for row in error_rows:
        source_index = int(row["source_index"])
        done[source_index] = str(row["source_identity"])
    for source_index, identity in done.items():
        if source_index >= len(records) or _source_identity(records[source_index]) != identity:
            raise ValueError(
                "Input records changed since the previous extraction run at "
                f"source_index={source_index}"
            )

    with binary_path.open("a+b") as binary:
        binary.truncate(committed_tokens * np.dtype("<u2").itemsize)
        binary.flush()
        os.fsync(binary.fileno())

    pending_indices = [
        index
        for index in range(args.shard, len(records), args.num_shards)
        if index not in done
    ]
    device = torch.device(args.device)
    loader = DataLoader(
        Subset(RecordDataset(records), pending_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=AudioCollator(max_audio_seconds, args.skip_audio_errors),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    backend = build_semantic_codec(
        cfg, model_dir=Path(cfg.paths.model_dir).expanduser().resolve()
    )
    if not isinstance(backend, MaskGCTSemanticCodec):
        raise TypeError("Expected MaskGCTSemanticCodec")
    backend = backend.to(device).eval()
    waveform_resampler = GPUWaveformResampler(device, target_rate=info.sample_rate)

    lookup = backend.codebook_lookup()
    _install_lookup_once(lookup_path, lookup)
    installed_payload = torch.load(lookup_path, map_location="cpu")
    installed_lookup = (
        installed_payload.get("lookup")
        if isinstance(installed_payload, dict)
        else installed_payload
    )
    if not isinstance(installed_lookup, torch.Tensor) or not torch.equal(
        installed_lookup.float(), lookup.float()
    ):
        raise ValueError(f"Existing lookup table does not match this codec: {lookup_path}")

    metadata = {
        **semantic_codec_info(cfg).to_dict(),
        "representation": "maskgct_codes",
        "codebook_size": int(lookup.size(0)),
        "codebook_dim": int(lookup.size(1)),
        "code_dtype": "uint16-le",
        "lookup_dtype": "float32",
        "lookup_file": lookup_path.name,
        "lookup_sha256": sha256_file(lookup_path),
        "checkpoint_path": str(backend.checkpoint_path),
        "checkpoint_sha256": sha256_file(backend.checkpoint_path),
        "max_audio_seconds": max_audio_seconds,
        "sources": [str(Path(source).expanduser()) for source in args.source],
        "records": len(records),
    }
    if metadata_path.is_file():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        stable_keys = (
            "fingerprint",
            "representation",
            "codebook_size",
            "codebook_dim",
            "code_dtype",
            "lookup_sha256",
            "checkpoint_sha256",
            "max_audio_seconds",
        )
        mismatches = [
            key
            for key in stable_keys
            if existing_metadata.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"Existing semantic code metadata is incompatible ({mismatches}): "
                f"{metadata_path}"
            )
    else:
        _atomic_write_json(metadata_path, metadata)

    extracted = 0
    failed = 0
    with (
        binary_path.open("r+b") as binary,
        manifest_path.open("a", encoding="utf-8") as manifest,
        errors_path.open("a", encoding="utf-8") as errors,
    ):
        binary.seek(0, os.SEEK_END)
        token_offset = binary.tell() // np.dtype("<u2").itemsize
        for valid_records, waveforms, sample_rates, failures in tqdm(
            loader,
            total=(len(pending_indices) + args.batch_size - 1) // args.batch_size,
            desc=suffix,
        ):
            for failure in failures:
                _append_jsonl(errors, failure)
                failed += 1
            if not waveforms:
                continue
            waveforms_16k = waveform_resampler(waveforms, sample_rates)
            codes_batch = backend.extract_codes(waveforms_16k)
            for record, codes in zip(valid_records, codes_batch, strict=True):
                codes = codes.detach().cpu().long().contiguous()
                if codes.ndim != 1 or codes.numel() == 0:
                    raise ValueError(
                        f"Invalid MaskGCT codes for {record['audio_path']}: {tuple(codes.shape)}"
                    )
                if int(codes.min()) < 0 or int(codes.max()) >= 8192:
                    raise ValueError(f"Out-of-range MaskGCT code for {record['audio_path']}")
                encoded = codes.numpy().astype("<u2", copy=False)
                binary.write(encoded.tobytes(order="C"))
                binary.flush()
                os.fsync(binary.fileno())

                entry = _public_record(record)
                entry.update(
                    {
                        "source_index": int(record["_source_index"]),
                        "source_identity": _source_identity(record),
                        "semantic_code_path": os.path.relpath(
                            binary_path, manifest_path.parent
                        ),
                        "semantic_code_offset": token_offset,
                        "semantic_code_length": int(codes.numel()),
                        "semantic_codebooks": 1,
                        "semantic_codec": info.name,
                        "semantic_dim": info.semantic_dim,
                        "semantic_fps": info.semantic_fps,
                        "semantic_fingerprint": metadata["fingerprint"],
                        "semantic_max_audio_seconds": max_audio_seconds,
                        "semantic_lookup_path": os.path.relpath(
                            lookup_path, manifest_path.parent
                        ),
                        "semantic_lookup_sha256": metadata["lookup_sha256"],
                    }
                )
                _append_jsonl(manifest, entry)
                token_offset += int(codes.numel())
                extracted += 1

    print(
        json.dumps(
            {
                "event": "complete",
                "shard": args.shard,
                "num_shards": args.num_shards,
                "records": len(records),
                "already_done": len(done),
                "extracted": extracted,
                "failed": failed,
                "manifest": str(manifest_path),
                "binary": str(binary_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
