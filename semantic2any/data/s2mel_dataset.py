from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tarfile
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


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


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


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
    max_prompt_seconds: float,
    min_generated_frames: int,
) -> int:
    min_frames = max(1, int(min_prompt_seconds * sample_rate / hop_length))
    max_frames = max(min_frames, int(max_prompt_seconds * sample_rate / hop_length))
    upper = max(1, min(max_frames, mel_len - min_generated_frames))
    if upper <= min_frames:
        return max(1, min(upper, mel_len - 1))
    return random.randint(min_frames, upper)


class S2MelJsonlDataset(Dataset):
    """JSONL manifest dataset for semantic2mel training."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).expanduser()
        self.base_dir = self.manifest_path.parent
        self.records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for key in ("audio_path", "mel_path", "semantic_path", "style_path"):
                    record[key] = _resolve_path(self.base_dir, record.get(key))
                record["_line_no"] = line_no
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


class S2MelCollator:
    def __init__(
        self,
        *,
        hop_length: int,
        sample_rate: int,
        min_prompt_seconds: float,
        max_prompt_seconds: float,
        min_generated_frames: int,
    ) -> None:
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.min_prompt_seconds = min_prompt_seconds
        self.max_prompt_seconds = max_prompt_seconds
        self.min_generated_frames = min_generated_frames

    def _collate_precomputed(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        mels = []
        semantics = []
        styles = []
        prompt_lens = []
        for record in records:
            mel = _load_tensor(record["mel_path"]).float()
            if mel.ndim == 3 and mel.size(0) == 1:
                mel = mel.squeeze(0)
            if mel.ndim != 2:
                raise ValueError(f"mel must be [80, T], got {tuple(mel.shape)}")
            semantic = _load_tensor(record["semantic_path"])
            style = _load_tensor(record["style_path"]).float().view(-1)
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
        has_precomputed = [
            bool(r.get("mel_path") and r.get("semantic_path") and r.get("style_path")) for r in records
        ]
        if all(has_precomputed):
            return self._collate_precomputed(records)
        if any(has_precomputed):
            raise ValueError("Do not mix precomputed and audio-only records in one batch")
        audio_paths = [r.get("audio_path") for r in records]
        if any(path is None for path in audio_paths):
            missing = [r.get("_line_no") for r, path in zip(records, audio_paths, strict=True) if path is None]
            raise ValueError(f"Records missing audio_path: {missing}")
        return {
            "audio_paths": audio_paths,
            "records": records,
            "is_precomputed": False,
        }
