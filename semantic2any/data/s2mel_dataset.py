from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tarfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

from semantic2any.data.prompt_bandwidth import simulate_lower_sample_rate
from semantic2any.third_party.indextts import mel_spectrogram


DEFAULT_MAX_AUDIO_SECONDS = 30.0
DEFAULT_MAX_PAIR_SECONDS = 50.0
DEFAULT_MAX_PROMPT_SECONDS = 20.0
DEFAULT_PROMPT_BANDWIDTH_AUG_PROB = 0.3
DEFAULT_PROMPT_BANDWIDTH_AUG_RATES = (16000, 22050)
_SEMANTIC_CODE_MEMMAPS: dict[str, np.memmap] = {}


def _load_tensor(path: str | Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("tensor", "mel", "semantic", "style", "data"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected tensor payload in {path}, got {type(obj)!r}")
    return obj


def _load_semantic_codes(record: dict[str, Any]) -> torch.Tensor:
    path = str(record["semantic_code_path"])
    offset = int(record["semantic_code_offset"])
    length = int(record["semantic_code_length"])
    if offset < 0 or length <= 0:
        raise ValueError(
            f"Invalid semantic code range offset={offset}, length={length}: {path}"
        )
    mmap = _SEMANTIC_CODE_MEMMAPS.get(path)
    if mmap is None:
        mmap = np.memmap(path, mode="r", dtype="<u2")
        _SEMANTIC_CODE_MEMMAPS[path] = mmap
    end = offset + length
    if end > mmap.size:
        raise ValueError(
            f"Semantic code range [{offset}, {end}) exceeds {path} ({mmap.size} tokens)"
        )
    return torch.from_numpy(np.array(mmap[offset:end], dtype=np.int64, copy=True))


def _pad_semantic_codes(records: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    codes = [_load_semantic_codes(record) for record in records]
    lengths = torch.tensor([item.numel() for item in codes], dtype=torch.long)
    return pad_sequence(codes, batch_first=True, padding_value=0), lengths


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def _has_semantic_codes(record: dict[str, Any]) -> bool:
    return bool(
        record.get("semantic_code_path")
        and record.get("semantic_code_length")
        and record.get("semantic_lookup_path")
        and record.get("semantic_lookup_sha256")
    )


def _is_gcs_uri(path: str) -> bool:
    return path.startswith("gs://")


def _strip_gcs_scheme(path: str) -> str:
    if not _is_gcs_uri(path):
        raise ValueError(f"Not a GCS URI: {path}")
    return path[5:]


def _join_uri(base: str, *parts: str) -> str:
    return "/".join([base.rstrip("/"), *(part.strip("/") for part in parts if part)])


def _path_name(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def _cache_key(*parts: str) -> str:
    hasher = hashlib.sha1()
    for part in parts:
        hasher.update(part.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def _locate_speechdata_audio(dataset_root: str, audio_path: str) -> tuple[str, str]:
    parts = audio_path.split("/")
    try:
        tar_index = next(index for index, part in enumerate(parts) if part.endswith(".tar"))
    except StopIteration as exc:
        raise ValueError(f"audio_path does not contain a tar shard: {audio_path}") from exc

    tar_rel_path = "/".join(parts[: tar_index + 1])
    member_name = "/".join(parts[tar_index + 1 :])
    if not member_name:
        raise ValueError(f"audio_path does not contain a tar member: {audio_path}")
    return _join_uri(dataset_root, tar_rel_path), member_name


def choose_prompt_len(
    mel_len: int,
    *,
    hop_length: int,
    sample_rate: int,
    min_prompt_seconds: float,
    max_prompt_seconds: float | None,
    min_generated_frames: int,
    min_target_seconds: float | None = None,
) -> int:
    min_frames = max(1, math.ceil(min_prompt_seconds * sample_rate / hop_length))
    max_frames = (
        mel_len
        if max_prompt_seconds is None
        else max(min_frames, int(max_prompt_seconds * sample_rate / hop_length))
    )
    min_target_frames = max(1, min_generated_frames)
    if min_target_seconds is not None:
        min_target_frames = max(
            min_target_frames,
            math.ceil(min_target_seconds * sample_rate / hop_length),
        )
    upper = min(max_frames, mel_len - min_target_frames)
    if upper < min_frames:
        required_seconds = min_prompt_seconds + (min_target_seconds or 0.0)
        raise ValueError(
            f"Audio has {mel_len} mel frames, too short for a "
            f"{min_prompt_seconds:g}s prompt and {min_target_seconds or 0:g}s target "
            f"(at least {required_seconds:g}s before STFT boundary effects)"
        )
    if upper == min_frames:
        return min_frames
    return random.randint(min_frames, upper)


def trim_paired_feature_lengths(
    prompt_mel_frames: int,
    target_mel_frames: int,
    prompt_semantic_frames: int,
    target_semantic_frames: int,
    *,
    hop_length: int,
    sample_rate: int,
    max_pair_seconds: float,
    min_prompt_seconds: float,
    min_generated_frames: int,
) -> tuple[int, int, int, int]:
    """Return prefix lengths after enforcing the paired-audio duration budget."""
    max_total_frames = max(1, int(max_pair_seconds * sample_rate / hop_length))
    min_prompt_frames = max(1, int(min_prompt_seconds * sample_rate / hop_length))
    if max_total_frames < min_prompt_frames + min_generated_frames:
        raise ValueError(
            "max_pair_seconds is too small for min_pair_prompt_seconds plus "
            "min_generated_frames"
        )
    if prompt_mel_frames < min_prompt_frames:
        raise ValueError(
            f"Prompt has {prompt_mel_frames} mel frames, fewer than the required "
            f"{min_prompt_frames}"
        )
    if target_mel_frames < min_generated_frames:
        raise ValueError(
            f"Target has {target_mel_frames} mel frames, fewer than the required "
            f"{min_generated_frames}"
        )
    if prompt_semantic_frames <= 0 or target_semantic_frames <= 0:
        raise ValueError("Prompt and target semantic features must not be empty")

    original_prompt_frames = prompt_mel_frames
    original_target_frames = target_mel_frames
    excess = max(0, prompt_mel_frames + target_mel_frames - max_total_frames)

    prompt_trim = min(excess, prompt_mel_frames - min_prompt_frames)
    prompt_mel_frames -= prompt_trim
    excess -= prompt_trim
    if excess:
        target_mel_frames -= excess
    if target_mel_frames < min_generated_frames:
        raise ValueError(
            "Paired sample cannot fit the duration budget while retaining the "
            "minimum prompt and target lengths"
        )

    def scaled_semantic_length(length: int, kept_mel: int, original_mel: int) -> int:
        if kept_mel >= original_mel:
            return length
        return max(1, min(length, round(length * kept_mel / original_mel)))

    return (
        prompt_mel_frames,
        target_mel_frames,
        scaled_semantic_length(
            prompt_semantic_frames, prompt_mel_frames, original_prompt_frames
        ),
        scaled_semantic_length(
            target_semantic_frames, target_mel_frames, original_target_frames
        ),
    )


class S2MelJsonlDataset(Dataset):
    """JSONL manifest dataset for semantic2mel training."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).expanduser()
        if self.manifest_path.is_dir():
            self.manifest_paths = sorted(self.manifest_path.glob("*.jsonl"))
            self.base_dir = self.manifest_path
        elif self.manifest_path.is_file():
            self.manifest_paths = [self.manifest_path]
            self.base_dir = self.manifest_path.parent
        else:
            raise FileNotFoundError(f"JSONL manifest source not found: {self.manifest_path}")
        if not self.manifest_paths:
            raise ValueError(f"No JSONL manifests found under {self.manifest_path}")

        self.records: list[dict[str, Any]] = []
        for current_manifest in self.manifest_paths:
            base_dir = current_manifest.parent
            with current_manifest.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    for key in (
                        "audio_path",
                        "mel_path",
                        "semantic_path",
                        "style_path",
                        "semantic_code_path",
                        "semantic_lookup_path",
                    ):
                        record[key] = _resolve_path(base_dir, record.get(key))
                    record["_line_no"] = line_no
                    record["_manifest_path"] = str(current_manifest)
                    self.records.append(record)
        if not self.records:
            raise ValueError(f"No records found in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.records[index])


class S2MelSpeechDataDataset(Dataset):
    """SpeechData normalized shards: metadata JSONL plus FLAC files inside tar shards."""

    def __init__(
        self,
        source: str | Path,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.source = str(source).rstrip("/")
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("S2MEL_SPEECHDATA_CACHE_DIR")
            or "/tmp/semantic2any-speechdata"
        ).expanduser()
        self._fs = None
        self.records: list[dict[str, Any]] = []

        for dataset_root, metadata_path in self._metadata_paths(self.source):
            self._load_metadata_shard(dataset_root, metadata_path)

        if not self.records:
            raise ValueError(f"No SpeechData records found in {self.source}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fs"] = None
        return state

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        audio_path = record.get("_speechdata_audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(
                f"SpeechData record missing audio_path: {record.get('_metadata_path')} line "
                f"{record.get('_line_no')}"
            )
        record["audio_path"] = self._materialize_audio(record["_speechdata_root"], audio_path)
        return record

    def _get_fs(self):
        if self._fs is None:
            try:
                import gcsfs
            except ImportError as exc:
                raise ImportError(
                    "Reading gs:// SpeechData sources requires gcsfs. Install it with "
                    "`uv pip install gcsfs`."
                ) from exc
            self._fs = gcsfs.GCSFileSystem()
        return self._fs

    def _open_binary(self, path: str):
        if _is_gcs_uri(path):
            return self._get_fs().open(_strip_gcs_scheme(path), "rb")
        return open(Path(path).expanduser(), "rb")

    def _metadata_paths(self, source: str) -> list[tuple[str, str]]:
        if _is_gcs_uri(source):
            return self._gcs_metadata_paths(source)
        return self._local_metadata_paths(Path(source).expanduser())

    def _local_metadata_paths(self, source: Path) -> list[tuple[str, str]]:
        if source.is_file():
            dataset_root = source.parent.parent if source.parent.name == "metadata" else source.parent
            return [(str(dataset_root), str(source))]

        if not source.is_dir():
            raise FileNotFoundError(f"SpeechData source not found: {source}")

        if (source / "metadata").is_dir():
            dataset_root = source
            metadata_dir = source / "metadata"
        elif source.name == "metadata":
            dataset_root = source.parent
            metadata_dir = source
        else:
            raise FileNotFoundError(
                f"SpeechData source must be a dataset dir, metadata dir, or JSONL shard: {source}"
            )

        metadata_paths = sorted(metadata_dir.glob("*.jsonl"))
        return [(str(dataset_root), str(path)) for path in metadata_paths]

    def _gcs_metadata_paths(self, source: str) -> list[tuple[str, str]]:
        if source.endswith(".jsonl"):
            metadata_marker = "/metadata/"
            if metadata_marker in source:
                dataset_root = source.split(metadata_marker, 1)[0]
            else:
                dataset_root = source.rsplit("/", 1)[0]
            return [(dataset_root, source)]

        if source.rstrip("/").endswith("/metadata"):
            dataset_root = source.rsplit("/", 1)[0]
            metadata_prefix = source.rstrip("/")
        else:
            dataset_root = source.rstrip("/")
            metadata_prefix = _join_uri(dataset_root, "metadata")

        matches = self._get_fs().glob(f"{_strip_gcs_scheme(metadata_prefix)}/*.jsonl")
        return [(dataset_root, f"gs://{path}") for path in sorted(matches)]

    def _load_metadata_shard(self, dataset_root: str, metadata_path: str) -> None:
        with self._open_binary(metadata_path) as file_obj:
            for line_no, raw_line in enumerate(file_obj, start=1):
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                record = json.loads(line)
                record["_line_no"] = line_no
                record["_metadata_path"] = metadata_path
                record["_speechdata_root"] = dataset_root
                record["_speechdata_audio_path"] = record.get("audio_path")
                self.records.append(record)

    def _materialize_audio(self, dataset_root: str, audio_path: str) -> str:
        tar_path, member_name = _locate_speechdata_audio(dataset_root, audio_path)
        suffix = Path(member_name).suffix or ".flac"
        key = _cache_key(tar_path, member_name)
        target = self.cache_dir / "audio" / key[:2] / f"{key}{suffix}"
        if target.is_file() and target.stat().st_size > 0:
            return str(target)

        target.parent.mkdir(parents=True, exist_ok=True)
        local_tar_path = self._local_tar_path(tar_path)
        tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with tarfile.open(local_tar_path, "r:*") as tar:
                member = tar.extractfile(member_name)
                if member is None:
                    raise FileNotFoundError(f"Tar member not found: {member_name} in {tar_path}")
                with member, tmp_path.open("wb") as out_file:
                    shutil.copyfileobj(member, out_file)
            tmp_path.replace(target)
        finally:
            tmp_path.unlink(missing_ok=True)
        return str(target)

    def _local_tar_path(self, tar_path: str) -> str:
        if not _is_gcs_uri(tar_path):
            return str(Path(tar_path).expanduser())

        tar_name = _path_name(tar_path)
        key = _cache_key(tar_path)
        target = self.cache_dir / "tars" / key[:2] / f"{key}-{tar_name}"
        if target.is_file() and target.stat().st_size > 0:
            return str(target)

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            with self._open_binary(tar_path) as in_file, tmp_path.open("wb") as out_file:
                shutil.copyfileobj(in_file, out_file)
            tmp_path.replace(target)
        finally:
            tmp_path.unlink(missing_ok=True)
        return str(target)


class S2MelInMemoryDataset(Dataset):
    """Precomputed s2mel features retained in CPU memory."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("In-memory s2mel dataset must not be empty")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _dataset_records(dataset: Dataset) -> list[dict[str, Any]]:
    records = getattr(dataset, "records", None)
    if not isinstance(records, list):
        raise TypeError("Speaker pairing requires a dataset with an in-memory records list")
    return records


def _record_identity(record: dict[str, Any], index: int) -> str:
    for key in ("id", "_speechdata_audio_path", "audio_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    return f"index:{index}"


def _record_can_be_prompt(
    record: dict[str, Any],
    *,
    min_prompt_seconds: float,
    hop_length: int,
    sample_rate: int,
) -> bool:
    mel = record.get("mel")
    if isinstance(mel, torch.Tensor):
        min_frames = max(1, int(min_prompt_seconds * sample_rate / hop_length))
        return mel.size(-1) >= min_frames
    duration = record.get("duration")
    if isinstance(duration, (int, float)):
        return float(duration) >= min_prompt_seconds
    # Generic manifests do not require duration. The collator performs the
    # authoritative frame-level check once their features are available.
    return True


def _record_has_singleton_semantic_budget(
    record: dict[str, Any],
    *,
    min_prompt_seconds: float,
    min_target_seconds: float,
) -> bool:
    """Precomputed code must be long enough for a frame-aligned singleton split."""
    code_length = record.get("semantic_code_length")
    if not isinstance(code_length, (int, float)):
        return True
    semantic_fps = record.get("semantic_fps", 50.0)
    if not isinstance(semantic_fps, (int, float)) or semantic_fps <= 0:
        return True
    min_code_frames = math.ceil((min_prompt_seconds + min_target_seconds) * float(semantic_fps))
    return int(code_length) >= min_code_frames


class S2MelSpeakerPairedDataset(Dataset):
    """Build same-speaker pairs, falling back to an aligned singleton split."""

    def __init__(
        self,
        target_dataset: Dataset,
        *,
        min_prompt_seconds: float,
        max_prompt_seconds: float,
        min_target_seconds: float,
        max_target_seconds: float,
        hop_length: int,
        sample_rate: int,
        seed: int = 0,
    ) -> None:
        self.target_dataset = target_dataset
        target_records = _dataset_records(target_dataset)
        if min_prompt_seconds <= 0 or min_target_seconds <= 0:
            raise ValueError("Minimum prompt and target durations must be positive")
        if max_prompt_seconds < min_prompt_seconds:
            raise ValueError("max_prompt_seconds must be >= min_prompt_seconds")
        if max_target_seconds < min_target_seconds:
            raise ValueError("max_target_seconds must be >= min_target_seconds")

        prompt_groups: dict[str, list[int]] = defaultdict(list)
        prompt_identities: dict[str, set[str]] = defaultdict(set)
        self.missing_speaker_count = 0
        self.missing_duration_count = 0
        for index, record in enumerate(target_records):
            speaker_id = record.get("speaker_id")
            if not isinstance(speaker_id, str) or not speaker_id:
                self.missing_speaker_count += 1
                continue
            duration = record.get("duration")
            if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)):
                self.missing_duration_count += 1
                continue
            if float(duration) >= min_prompt_seconds:
                identity = _record_identity(record, index)
                prompt_groups[speaker_id].append(index)
                prompt_identities[speaker_id].add(identity)

        self.target_indices: list[int] = []
        self.target_speaker_ids: list[str] = []
        self.singleton_splits: list[bool] = []
        self.paired_target_count = 0
        self.singleton_target_count = 0
        self.too_short_target_count = 0
        self.overlong_target_count = 0
        self.unusable_target_count = 0
        for target_index, target_record in enumerate(target_records):
            speaker_id = target_record.get("speaker_id")
            if not isinstance(speaker_id, str) or not speaker_id:
                continue
            duration = target_record.get("duration")
            if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)):
                continue
            duration = float(duration)
            if duration < min_target_seconds:
                self.too_short_target_count += 1
                continue
            if duration > max_target_seconds:
                self.overlong_target_count += 1
                continue
            target_identity = _record_identity(target_record, target_index)
            identities = prompt_identities.get(speaker_id, set())
            singleton_split = not identities or identities == {target_identity}
            if singleton_split:
                if duration < min_prompt_seconds + min_target_seconds:
                    self.unusable_target_count += 1
                    continue
                if not _record_has_singleton_semantic_budget(
                    target_record,
                    min_prompt_seconds=min_prompt_seconds,
                    min_target_seconds=min_target_seconds,
                ):
                    self.unusable_target_count += 1
                    continue
            self.target_indices.append(target_index)
            self.target_speaker_ids.append(speaker_id)
            self.singleton_splits.append(singleton_split)
            self.singleton_target_count += int(singleton_split)
            self.paired_target_count += int(not singleton_split)

        if not self.target_indices:
            raise ValueError(
                "No usable speaker-conditioned samples are available. Targets need "
                f"speaker_id and duration in [{min_target_seconds:g}, "
                f"{max_target_seconds:g}] seconds; singleton utterances must also fit "
                f"a {min_prompt_seconds:g}-second prompt."
            )
        self.min_prompt_seconds = float(min_prompt_seconds)
        self.max_prompt_seconds = float(max_prompt_seconds)
        self.min_target_seconds = float(min_target_seconds)
        self.max_target_seconds = float(max_target_seconds)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.prompt_groups = dict(prompt_groups)
        self.target_records = target_records
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.target_indices)

    def _select_prompt_index(self, index: int, target_index: int) -> int:
        target_identity = _record_identity(
            self.target_records[target_index],
            target_index,
        )
        group = self.prompt_groups[self.target_speaker_ids[index]]
        rng = random.Random(self.seed + index)
        start = rng.randrange(len(group))
        for offset in range(len(group)):
            candidate_index = group[(start + offset) % len(group)]
            if (
                _record_identity(
                    self.target_records[candidate_index],
                    candidate_index,
                )
                != target_identity
            ):
                return candidate_index
        raise RuntimeError("Paired sample has no distinct same-speaker prompt")

    def estimated_sample_seconds(self, index: int) -> float:
        target_index = self.target_indices[index]
        target_record = self.target_records[target_index]
        duration = float(target_record.get("duration", self.max_target_seconds))
        if self.singleton_splits[index]:
            return duration
        prompt_index = self._select_prompt_index(index, target_index)
        prompt_duration = float(
            self.target_records[prompt_index].get("duration", self.max_prompt_seconds)
        )
        return duration + min(prompt_duration, self.max_prompt_seconds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_index = self.target_indices[index]
        singleton_split = self.singleton_splits[index]
        target = self.target_dataset[target_index]
        if singleton_split:
            prompt = target
        else:
            prompt_index = self._select_prompt_index(index, target_index)
            prompt = self.target_dataset[prompt_index]
        return {
            "prompt": prompt,
            "target": target,
            "speaker_id": target["speaker_id"],
            "singleton_split": singleton_split,
        }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Yield same-bucket batch groups for Accelerate/DDP sharding."""

    def __init__(
        self,
        lengths: list[float],
        *,
        batch_size: int,
        world_size: int,
        boundaries: list[float],
        seed: int = 0,
        drop_last: bool = True,
        shuffle: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.lengths = [float(length) for length in lengths]
        self.batch_size = int(batch_size)
        self.world_size = int(world_size)
        self.boundaries = sorted(float(boundary) for boundary in boundaries)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self._length = len(self._build_batches())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _bucket_id(self, length: float) -> int:
        for index, boundary in enumerate(self.boundaries):
            if length <= boundary:
                return index
        return len(self.boundaries)

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, length in enumerate(self.lengths):
            buckets[self._bucket_id(length)].append(index)

        groups: list[list[list[int]]] = []
        for bucket_indices in buckets.values():
            indices = list(bucket_indices)
            if self.shuffle:
                rng.shuffle(indices)
            local_batches = [
                indices[start : start + self.batch_size]
                for start in range(0, len(indices), self.batch_size)
            ]
            if self.drop_last:
                local_batches = [
                    batch for batch in local_batches if len(batch) == self.batch_size
                ]
            else:
                local_batches = [batch for batch in local_batches if batch]
            usable = (len(local_batches) // self.world_size) * self.world_size
            local_batches = local_batches[:usable]
            for start in range(0, len(local_batches), self.world_size):
                groups.append(local_batches[start : start + self.world_size])

        if self.shuffle:
            rng.shuffle(groups)
        return [batch for group in groups for batch in group]

    def __iter__(self):
        yield from self._build_batches()

    def __len__(self) -> int:
        return self._length


def _normalize_mel(mel: torch.Tensor) -> torch.Tensor:
    mel = mel.float()
    if mel.ndim == 3 and mel.size(0) == 1:
        mel = mel.squeeze(0)
    if mel.ndim != 2:
        raise ValueError(f"mel must be [channels, T], got {tuple(mel.shape)}")
    return mel


def _normalize_semantic(semantic: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(semantic):
        if semantic.ndim == 3 and semantic.size(0) == 1:
            semantic = semantic.squeeze(0)
        if semantic.ndim != 2:
            raise ValueError(f"Continuous semantic must be [T, C], got {tuple(semantic.shape)}")
        return semantic
    if semantic.ndim == 1:
        semantic = semantic.unsqueeze(0)
    if semantic.ndim != 2:
        raise ValueError(f"Discrete semantic must be [Q, T], got {tuple(semantic.shape)}")
    return semantic


def _semantic_length(semantic: torch.Tensor) -> int:
    return semantic.size(0) if torch.is_floating_point(semantic) else semantic.size(-1)


def _load_record_features(record: dict[str, Any]) -> dict[str, torch.Tensor]:
    mel_value = record.get("mel")
    semantic_value = record.get("semantic")
    style_value = record.get("style")
    mel = mel_value if isinstance(mel_value, torch.Tensor) else _load_tensor(record["mel_path"])
    semantic = (
        semantic_value
        if isinstance(semantic_value, torch.Tensor)
        else _load_tensor(record["semantic_path"])
    )
    style = style_value if isinstance(style_value, torch.Tensor) else _load_tensor(record["style_path"])
    style = style.float().view(-1)
    if style.numel() != 192:
        raise ValueError(f"style must contain 192 values, got {style.numel()}")
    return {
        "mel": _normalize_mel(mel),
        "semantic": _normalize_semantic(semantic),
        "style": style,
    }


def collate_paired_features(
    prompt_features: list[dict[str, torch.Tensor]],
    target_features: list[dict[str, torch.Tensor]],
    *,
    hop_length: int,
    sample_rate: int,
    max_pair_seconds: float,
    min_prompt_seconds: float,
    min_generated_frames: int,
    records: list[dict[str, Any]] | None = None,
    is_precomputed: bool = False,
) -> dict[str, Any]:
    """Assemble [prompt, target] timelines using prompt style and both semantics."""
    if not prompt_features or len(prompt_features) != len(target_features):
        raise ValueError("Prompt and target feature lists must have the same non-zero length")

    mels = []
    semantics = []
    styles = []
    prompt_lens = []
    prompt_semantic_lens = []
    for prompt_item, target_item in zip(prompt_features, target_features, strict=True):
        prompt_mel = _normalize_mel(prompt_item["mel"])
        target_mel = _normalize_mel(target_item["mel"])
        prompt_semantic = _normalize_semantic(prompt_item["semantic"])
        target_semantic = _normalize_semantic(target_item["semantic"])
        if torch.is_floating_point(prompt_semantic) != torch.is_floating_point(target_semantic):
            raise TypeError("Prompt and target semantic features must use the same representation")

        prompt_keep, target_keep, prompt_semantic_keep, target_semantic_keep = (
            trim_paired_feature_lengths(
                prompt_mel.size(-1),
                target_mel.size(-1),
                _semantic_length(prompt_semantic),
                _semantic_length(target_semantic),
                hop_length=hop_length,
                sample_rate=sample_rate,
                max_pair_seconds=max_pair_seconds,
                min_prompt_seconds=min_prompt_seconds,
                min_generated_frames=min_generated_frames,
            )
        )
        paired_mel = torch.cat(
            [prompt_mel[:, :prompt_keep], target_mel[:, :target_keep]], dim=-1
        )
        if torch.is_floating_point(prompt_semantic):
            paired_semantic = torch.cat(
                [
                    prompt_semantic[:prompt_semantic_keep],
                    target_semantic[:target_semantic_keep],
                ],
                dim=0,
            )
        else:
            if prompt_semantic.size(0) != target_semantic.size(0):
                raise ValueError("Prompt and target must have the same number of codebooks")
            paired_semantic = torch.cat(
                [
                    prompt_semantic[:, :prompt_semantic_keep],
                    target_semantic[:, :target_semantic_keep],
                ],
                dim=-1,
            )

        style = prompt_item["style"].float().view(-1)
        if style.numel() != 192:
            raise ValueError(f"style must contain 192 values, got {style.numel()}")
        mels.append(paired_mel.transpose(0, 1))
        semantics.append(paired_semantic)
        styles.append(style)
        prompt_lens.append(prompt_keep)
        prompt_semantic_lens.append(prompt_semantic_keep)

    device = mels[0].device
    mel_lens = torch.tensor([x.size(0) for x in mels], dtype=torch.long, device=device)
    mel = pad_sequence(mels, batch_first=True, padding_value=0.0).transpose(1, 2)
    if torch.is_floating_point(semantics[0]):
        semantic_lens = torch.tensor(
            [x.size(0) for x in semantics], dtype=torch.long, device=device
        )
        semantic = pad_sequence(
            [x.float() for x in semantics], batch_first=True, padding_value=0.0
        )
    else:
        q = semantics[0].size(0)
        sem_lens = [x.size(-1) for x in semantics]
        semantic_lens = torch.tensor(sem_lens, dtype=torch.long, device=device)
        semantic = torch.zeros(
            len(semantics), q, max(sem_lens), dtype=torch.long, device=device
        )
        for index, item in enumerate(semantics):
            semantic[index, :, : item.size(-1)] = item.long()

    batch: dict[str, Any] = {
        "mel": mel,
        "mel_lens": mel_lens,
        "semantic": semantic,
        "semantic_lens": semantic_lens,
        "style": torch.stack(styles),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long, device=device),
        "prompt_semantic_lens": torch.tensor(
            prompt_semantic_lens, dtype=torch.long, device=device
        ),
        "is_precomputed": is_precomputed,
        "is_paired": True,
    }
    if records is not None:
        batch["records"] = records
    return batch


class S2MelCollator:
    def __init__(
        self,
        *,
        hop_length: int,
        sample_rate: int,
        min_prompt_seconds: float,
        max_prompt_seconds: float | None,
        min_generated_frames: int,
        min_target_seconds: float | None = None,
        max_target_seconds: float | None = None,
        max_pair_seconds: float = DEFAULT_MAX_PAIR_SECONDS,
        min_pair_prompt_seconds: float = 3.0,
        decode_audio_in_worker: bool = False,
        skip_audio_errors: bool = False,
        max_audio_seconds: float | None = DEFAULT_MAX_AUDIO_SECONDS,
        expected_semantic_codec: str | None = None,
        expected_semantic_fingerprint: str | None = None,
        extract_mel_in_worker: bool = False,
        mel_n_fft: int = 2048,
        mel_win_length: int = 2048,
        mel_n_mels: int = 128,
        mel_fmin: float = 0.0,
        mel_fmax: float | None = None,
        prompt_bandwidth_aug_prob: float = DEFAULT_PROMPT_BANDWIDTH_AUG_PROB,
        prompt_bandwidth_aug_rates: tuple[int, ...] = DEFAULT_PROMPT_BANDWIDTH_AUG_RATES,
    ) -> None:
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.min_prompt_seconds = min_prompt_seconds
        self.max_prompt_seconds = max_prompt_seconds
        self.min_generated_frames = min_generated_frames
        self.min_target_seconds = min_target_seconds
        self.max_target_seconds = max_target_seconds
        self.max_pair_seconds = max_pair_seconds
        self.min_pair_prompt_seconds = min_pair_prompt_seconds
        self.decode_audio_in_worker = decode_audio_in_worker
        self.skip_audio_errors = skip_audio_errors
        self.max_audio_seconds = max_audio_seconds
        self.expected_semantic_codec = expected_semantic_codec
        self.expected_semantic_fingerprint = expected_semantic_fingerprint
        self.extract_mel_in_worker = extract_mel_in_worker
        self.mel_args = {
            "n_fft": int(mel_n_fft),
            "num_mels": int(mel_n_mels),
            "sampling_rate": int(sample_rate),
            "hop_size": int(hop_length),
            "win_size": int(mel_win_length),
            "fmin": float(mel_fmin),
            "fmax": mel_fmax,
            "center": False,
        }
        self.prompt_bandwidth_aug_prob = float(prompt_bandwidth_aug_prob)
        self.prompt_bandwidth_aug_rates = tuple(int(rate) for rate in prompt_bandwidth_aug_rates)

    def _validate_precomputed_metadata(self, records: list[dict[str, Any]]) -> None:
        if self.expected_semantic_codec is None:
            return
        flattened = []
        for record in records:
            if isinstance(record.get("prompt"), dict):
                flattened.extend((record["prompt"], record["target"]))
            else:
                flattened.append(record)
        for record in flattened:
            actual_codec = record.get("semantic_codec")
            actual_fingerprint = record.get("semantic_fingerprint")
            if actual_codec is not None and actual_codec != self.expected_semantic_codec:
                raise ValueError(
                    "Precomputed semantic codec mismatch: "
                    f"manifest={actual_codec}, config={self.expected_semantic_codec}"
                )
            if (
                actual_fingerprint is not None
                and self.expected_semantic_fingerprint is not None
                and actual_fingerprint != self.expected_semantic_fingerprint
            ):
                raise ValueError(
                    "Precomputed semantic fingerprint mismatch: "
                    f"manifest={actual_fingerprint}, "
                    f"config={self.expected_semantic_fingerprint}"
                )

    def _decode_audio_paths(
        self,
        audio_paths: list[str],
        *,
        max_audio_seconds: float | None,
    ) -> tuple[list[torch.Tensor], list[int], list[int]]:
        waveforms = []
        sample_rates = []
        valid_indices = []
        decoded: dict[str, tuple[torch.Tensor, int] | None] = {}
        for index, path in enumerate(audio_paths):
            cached = decoded.get(path)
            if path not in decoded:
                try:
                    cached = torchaudio.load(path)
                except (OSError, RuntimeError, ValueError) as exc:
                    if not self.skip_audio_errors:
                        raise
                    warnings.warn(
                        f"Skipping undecodable audio file {path}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    cached = None
                decoded[path] = cached
            if cached is None:
                continue
            audio, sample_rate = cached
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if max_audio_seconds is not None:
                max_samples = int(max_audio_seconds * sample_rate)
                audio = audio[:, :max_samples]
            waveforms.append(audio)
            sample_rates.append(sample_rate)
            valid_indices.append(index)
        return waveforms, sample_rates, valid_indices

    def _resample_to_mel_rate(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        sample_rate = int(sample_rate)
        if sample_rate == self.sample_rate:
            return waveform
        return torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

    def _maybe_limit_prompt_bandwidth(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.prompt_bandwidth_aug_prob <= 0.0 or not self.prompt_bandwidth_aug_rates:
            return waveform
        if random.random() >= self.prompt_bandwidth_aug_prob:
            return waveform
        rate = random.choice(self.prompt_bandwidth_aug_rates)
        return simulate_lower_sample_rate(waveform, self.sample_rate, rate)

    def _mel_from_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        return mel_spectrogram(waveform.float(), **self.mel_args).squeeze(0).cpu()

    def _attach_paired_worker_features(
        self,
        batch: dict[str, Any],
        *,
        prompt_waveforms: list[torch.Tensor],
        prompt_sample_rates: list[int],
        target_waveforms: list[torch.Tensor],
        target_sample_rates: list[int],
    ) -> None:
        if self.min_target_seconds is None or self.max_target_seconds is None:
            raise ValueError("Worker paired mel extraction requires target duration bounds")
        if self.max_prompt_seconds is None:
            raise ValueError("Worker paired mel extraction requires max_prompt_seconds")
        singleton_splits = [bool(value) for value in batch["singleton_splits"]]
        if not (
            len(prompt_waveforms)
            == len(prompt_sample_rates)
            == len(target_waveforms)
            == len(target_sample_rates)
            == len(singleton_splits)
        ):
            raise ValueError("Decoded paired audio fields must have matching lengths")

        prompt_codes = batch["prompt_semantic_codes"]
        prompt_code_lens = batch["prompt_semantic_code_lens"]
        target_codes = batch["target_semantic_codes"]
        target_code_lens = batch["target_semantic_code_lens"]
        prompt_features: list[dict[str, torch.Tensor]] = []
        target_features: list[dict[str, torch.Tensor]] = []
        style = torch.zeros(192, dtype=torch.float32)

        for index, singleton_split in enumerate(singleton_splits):
            if singleton_split:
                source = target_waveforms[index]
                source_rate = int(target_sample_rates[index])
                source_samples = source.size(-1)
                min_prompt_samples = math.ceil(self.min_pair_prompt_seconds * source_rate)
                min_target_samples = math.ceil(self.min_target_seconds * source_rate)
                max_prompt_samples = int(self.max_prompt_seconds * source_rate)
                lower_sample = min_prompt_samples
                upper_sample = min(max_prompt_samples, source_samples - min_target_samples)
                if upper_sample < lower_sample:
                    raise ValueError("Singleton audio is too short for worker-side paired mel extraction")

                code_length = int(target_code_lens[index])
                lower_code = max(1, math.ceil(lower_sample * code_length / source_samples))
                upper_code = min(
                    code_length - 1,
                    math.floor(upper_sample * code_length / source_samples),
                )
                if upper_code < lower_code:
                    raise ValueError("Singleton audio has too few semantic frames for worker extraction")
                split_code = lower_code if upper_code == lower_code else random.randint(lower_code, upper_code)
                split_sample = round(split_code * source_samples / code_length)
                split_sample = max(lower_sample, min(upper_sample, split_sample))

                prompt_segment = source[:, :split_sample]
                target_segment = source[:, split_sample:]
                prompt_code_ids = target_codes[index, :split_code]
                target_code_ids = target_codes[index, split_code:code_length]
                prompt_rate = source_rate
                target_rate = source_rate
            else:
                prompt_source = prompt_waveforms[index]
                prompt_rate = int(prompt_sample_rates[index])
                target_segment = target_waveforms[index]
                target_rate = int(target_sample_rates[index])
                prompt_samples = min(
                    prompt_source.size(-1),
                    int(self.max_prompt_seconds * prompt_rate),
                )
                if prompt_samples < math.ceil(self.min_pair_prompt_seconds * prompt_rate):
                    raise ValueError("Prompt audio is too short for worker-side paired mel extraction")
                target_seconds = target_segment.size(-1) / target_rate
                if target_seconds < self.min_target_seconds:
                    raise ValueError("Target audio is too short for worker-side paired mel extraction")
                if target_seconds > self.max_target_seconds + (1.0 / target_rate):
                    raise ValueError("Target audio exceeds worker-side paired mel duration limit")

                prompt_segment = prompt_source[:, :prompt_samples]
                full_prompt_len = int(prompt_code_lens[index])
                prompt_keep = max(
                    1,
                    min(
                        full_prompt_len,
                        round(full_prompt_len * prompt_samples / prompt_source.size(-1)),
                    ),
                )
                prompt_code_ids = prompt_codes[index, :prompt_keep]
                target_code_ids = target_codes[index, : int(target_code_lens[index])]

            prompt_mel_waveform = self._resample_to_mel_rate(prompt_segment, prompt_rate)
            prompt_mel_waveform = self._maybe_limit_prompt_bandwidth(prompt_mel_waveform)
            target_mel_waveform = self._resample_to_mel_rate(target_segment, target_rate)
            prompt_features.append(
                {
                    "mel": self._mel_from_waveform(prompt_mel_waveform),
                    "semantic": prompt_code_ids.long(),
                    "style": style,
                }
            )
            target_features.append(
                {
                    "mel": self._mel_from_waveform(target_mel_waveform),
                    "semantic": target_code_ids.long(),
                    "style": style,
                }
            )

        worker_batch = collate_paired_features(
            prompt_features,
            target_features,
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            max_pair_seconds=self.max_prompt_seconds + self.max_target_seconds,
            min_prompt_seconds=self.min_pair_prompt_seconds,
            min_generated_frames=self.min_generated_frames,
            records=batch.get("records"),
            is_precomputed=False,
        )
        batch.update(worker_batch)
        batch["worker_precomputed_mel"] = True

    @staticmethod
    def _has_precomputed(record: dict[str, Any]) -> bool:
        return (
            all(isinstance(record.get(key), torch.Tensor) for key in ("mel", "semantic", "style"))
            or bool(
                record.get("mel_path")
                and record.get("semantic_path")
                and record.get("style_path")
            )
        )

    @staticmethod
    def _has_semantic_codes(record: dict[str, Any]) -> bool:
        return _has_semantic_codes(record)

    def _semantic_code_batch_metadata(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, str]:
        lookup_paths = {str(record["semantic_lookup_path"]) for record in records}
        lookup_hashes = {str(record["semantic_lookup_sha256"]) for record in records}
        fingerprints = {str(record.get("semantic_fingerprint", "")) for record in records}
        codecs = {str(record.get("semantic_codec", "")) for record in records}
        if len(lookup_paths) != 1 or len(lookup_hashes) != 1:
            raise ValueError("A batch must use one semantic lookup table and checksum")
        if len(fingerprints) != 1 or len(codecs) != 1 or codecs != {"maskgct"}:
            raise ValueError("A semantic code batch must use one MaskGCT fingerprint")
        encoded_max_durations = {
            float(record["semantic_max_audio_seconds"])
            for record in records
            if record.get("semantic_max_audio_seconds") is not None
        }
        if len(encoded_max_durations) > 1:
            raise ValueError("A batch must use one semantic max-audio duration")
        if (
            encoded_max_durations
            and self.max_audio_seconds is not None
            and not math.isclose(
                next(iter(encoded_max_durations)),
                self.max_audio_seconds,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                "Precomputed semantic code duration limit does not match training: "
                f"manifest={next(iter(encoded_max_durations))}, "
                f"config={self.max_audio_seconds}"
            )
        return {
            "semantic_lookup_path": next(iter(lookup_paths)),
            "semantic_lookup_sha256": next(iter(lookup_hashes)),
            "semantic_fingerprint": next(iter(fingerprints)),
        }

    def _attach_single_semantic_codes(
        self,
        batch: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        semantic_codes, semantic_code_lens = _pad_semantic_codes(records)
        batch.update(
            {
                "semantic_codes": semantic_codes,
                "semantic_code_lens": semantic_code_lens,
                "has_semantic_codes": True,
                **self._semantic_code_batch_metadata(records),
            }
        )

    def _attach_paired_semantic_codes(
        self,
        batch: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> None:
        prompt_records = [record["prompt"] for record in records]
        target_records = [record["target"] for record in records]
        flattened = prompt_records + target_records
        prompt_codes, prompt_code_lens = _pad_semantic_codes(prompt_records)
        target_codes, target_code_lens = _pad_semantic_codes(target_records)
        batch.update(
            {
                "prompt_semantic_codes": prompt_codes,
                "prompt_semantic_code_lens": prompt_code_lens,
                "target_semantic_codes": target_codes,
                "target_semantic_code_lens": target_code_lens,
                "has_semantic_codes": True,
                **self._semantic_code_batch_metadata(flattened),
            }
        )

    def _collate_paired_precomputed(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return collate_paired_features(
            [_load_record_features(record["prompt"]) for record in records],
            [_load_record_features(record["target"]) for record in records],
            hop_length=self.hop_length,
            sample_rate=self.sample_rate,
            max_pair_seconds=self.max_pair_seconds,
            min_prompt_seconds=self.min_pair_prompt_seconds,
            min_generated_frames=self.min_generated_frames,
            records=records,
            is_precomputed=True,
        )

    def _collate_precomputed(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        mels = []
        semantics = []
        styles = []
        prompt_lens = []
        for record in records:
            mel_value = record.get("mel")
            mel = mel_value.float() if isinstance(mel_value, torch.Tensor) else _load_tensor(record["mel_path"]).float()
            if mel.ndim == 3 and mel.size(0) == 1:
                mel = mel.squeeze(0)
            if mel.ndim != 2:
                raise ValueError(f"mel must be [80, T], got {tuple(mel.shape)}")
            semantic_value = record.get("semantic")
            semantic = (
                semantic_value
                if isinstance(semantic_value, torch.Tensor)
                else _load_tensor(record["semantic_path"])
            )
            style_value = record.get("style")
            style = (
                style_value.float().view(-1)
                if isinstance(style_value, torch.Tensor)
                else _load_tensor(record["style_path"]).float().view(-1)
            )
            if style.numel() != 192:
                raise ValueError(f"style must contain 192 values, got {style.numel()}")

            mel_len = mel.size(-1)
            prompt_len = int(
                record.get("prompt_len")
                or choose_prompt_len(
                    mel_len,
                    hop_length=self.hop_length,
                    sample_rate=self.sample_rate,
                    min_prompt_seconds=self.min_prompt_seconds,
                    max_prompt_seconds=self.max_prompt_seconds,
                    min_generated_frames=self.min_generated_frames,
                    min_target_seconds=self.min_target_seconds,
                )
            )
            mels.append(mel.transpose(0, 1))
            semantics.append(semantic)
            styles.append(style)
            prompt_lens.append(prompt_len)

        mel_lens = torch.tensor([x.size(0) for x in mels], dtype=torch.long)
        mel = pad_sequence(mels, batch_first=True, padding_value=0.0).transpose(1, 2)

        if torch.is_floating_point(semantics[0]):
            semantic_lens = torch.tensor([x.size(0) for x in semantics], dtype=torch.long)
            semantic = pad_sequence([x.float() for x in semantics], batch_first=True, padding_value=0.0)
        else:
            q = semantics[0].size(0) if semantics[0].ndim == 2 else 1
            sem_lens = [x.size(-1) for x in semantics]
            max_sem = max(sem_lens)
            semantic = torch.zeros(len(semantics), q, max_sem, dtype=torch.long)
            for idx, x in enumerate(semantics):
                if x.ndim == 1:
                    x = x.unsqueeze(0)
                semantic[idx, :, : x.size(-1)] = x.long()
            semantic_lens = torch.tensor(sem_lens, dtype=torch.long)

        return {
            "mel": mel,
            "mel_lens": mel_lens,
            "semantic": semantic,
            "semantic_lens": semantic_lens,
            "style": torch.stack(styles),
            "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
            "records": records,
            "is_precomputed": True,
        }

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        self._validate_precomputed_metadata(records)
        paired = [
            isinstance(record.get("prompt"), dict) and isinstance(record.get("target"), dict)
            for record in records
        ]
        if any(paired) and not all(paired):
            raise ValueError("Do not mix paired and single-utterance records in one batch")
        if all(paired):
            prompt_records = [record["prompt"] for record in records]
            target_records = [record["target"] for record in records]
            has_precomputed = [
                self._has_precomputed(prompt) and self._has_precomputed(target)
                for prompt, target in zip(prompt_records, target_records, strict=True)
            ]
            partially_precomputed = [
                self._has_precomputed(prompt) or self._has_precomputed(target)
                for prompt, target in zip(prompt_records, target_records, strict=True)
            ]
            has_semantic_codes = [
                self._has_semantic_codes(prompt) and self._has_semantic_codes(target)
                for prompt, target in zip(prompt_records, target_records, strict=True)
            ]
            partially_semantic_codes = [
                self._has_semantic_codes(prompt) or self._has_semantic_codes(target)
                for prompt, target in zip(prompt_records, target_records, strict=True)
            ]
            if all(has_precomputed):
                if any(partially_semantic_codes):
                    raise ValueError(
                        "Do not combine full precomputed features with semantic codes"
                    )
                return self._collate_paired_precomputed(records)
            if any(partially_precomputed):
                raise ValueError("Both sides of every pair must use the same feature mode")
            if any(partially_semantic_codes) and not all(has_semantic_codes):
                raise ValueError("Both sides of every pair must provide semantic codes")
            prompt_audio_paths = [record.get("audio_path") for record in prompt_records]
            target_audio_paths = [record.get("audio_path") for record in target_records]
            if any(path is None for path in prompt_audio_paths + target_audio_paths):
                raise ValueError("Paired records must contain prompt and target audio_path")
            batch = {
                "prompt_audio_paths": prompt_audio_paths,
                "target_audio_paths": target_audio_paths,
                "singleton_splits": [
                    bool(record.get("singleton_split", False)) for record in records
                ],
                "records": records,
                "is_precomputed": False,
                "is_paired": True,
            }
            if self.decode_audio_in_worker:
                prompt_waveforms, prompt_sample_rates, prompt_indices = self._decode_audio_paths(
                    prompt_audio_paths,
                    max_audio_seconds=self.max_audio_seconds,
                )
                target_waveforms, target_sample_rates, target_indices = self._decode_audio_paths(
                    target_audio_paths,
                    max_audio_seconds=None,
                )
                prompt_decoded = {
                    index: (waveform, sample_rate)
                    for index, waveform, sample_rate in zip(
                        prompt_indices, prompt_waveforms, prompt_sample_rates, strict=True
                    )
                }
                target_decoded = {
                    index: (waveform, sample_rate)
                    for index, waveform, sample_rate in zip(
                        target_indices, target_waveforms, target_sample_rates, strict=True
                    )
                }
                valid_indices = [
                    index
                    for index in range(len(records))
                    if index in prompt_decoded and index in target_decoded
                ]
                if not valid_indices:
                    raise RuntimeError("No fully decodable prompt/target pairs remain in batch")
                prompt_audio_paths = [prompt_audio_paths[index] for index in valid_indices]
                target_audio_paths = [target_audio_paths[index] for index in valid_indices]
                records = [records[index] for index in valid_indices]
                batch.update(
                    {
                        "prompt_audio_paths": prompt_audio_paths,
                        "target_audio_paths": target_audio_paths,
                        "singleton_splits": [
                            bool(records[index].get("singleton_split", False))
                            for index in range(len(records))
                        ],
                        "records": records,
                        "prompt_audio_waveforms": [
                            prompt_decoded[index][0] for index in valid_indices
                        ],
                        "prompt_audio_sample_rates": [
                            prompt_decoded[index][1] for index in valid_indices
                        ],
                        "target_audio_waveforms": [
                            target_decoded[index][0] for index in valid_indices
                        ],
                        "target_audio_sample_rates": [
                            target_decoded[index][1] for index in valid_indices
                        ],
                    }
                )
            if all(has_semantic_codes):
                self._attach_paired_semantic_codes(batch, records)
                if self.extract_mel_in_worker:
                    if not self.decode_audio_in_worker:
                        raise ValueError(
                            "extract_mel_in_worker requires decode_audio_in_worker=true"
                        )
                    self._attach_paired_worker_features(
                        batch,
                        prompt_waveforms=batch.pop("prompt_audio_waveforms"),
                        prompt_sample_rates=batch.pop("prompt_audio_sample_rates"),
                        target_waveforms=batch.pop("target_audio_waveforms"),
                        target_sample_rates=batch.pop("target_audio_sample_rates"),
                    )
            return batch

        has_precomputed = [self._has_precomputed(record) for record in records]
        has_semantic_codes = [self._has_semantic_codes(record) for record in records]
        if all(has_precomputed):
            if any(has_semantic_codes):
                raise ValueError(
                    "Do not combine full precomputed features with semantic codes"
                )
            return self._collate_precomputed(records)
        if any(has_precomputed):
            raise ValueError("Do not mix precomputed and audio-only records in one batch")
        if any(has_semantic_codes) and not all(has_semantic_codes):
            raise ValueError("Do not mix records with and without semantic codes")
        audio_paths = [r.get("audio_path") for r in records]
        if any(path is None for path in audio_paths):
            missing = [r.get("_line_no") for r, path in zip(records, audio_paths, strict=True) if path is None]
            raise ValueError(f"Records missing audio_path: {missing}")
        batch = {
            "audio_paths": audio_paths,
            "records": records,
            "is_precomputed": False,
        }
        if self.decode_audio_in_worker:
            waveforms, sample_rates, valid_indices = self._decode_audio_paths(
                audio_paths,
                max_audio_seconds=self.max_audio_seconds,
            )
            if not valid_indices:
                raise RuntimeError("No decodable audio files remain in batch")
            batch["audio_paths"] = [audio_paths[index] for index in valid_indices]
            batch["records"] = [records[index] for index in valid_indices]
            batch["audio_waveforms"] = waveforms
            batch["audio_sample_rates"] = sample_rates
        if all(has_semantic_codes):
            self._attach_single_semantic_codes(batch, batch["records"])
        return batch
