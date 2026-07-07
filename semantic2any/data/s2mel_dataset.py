from __future__ import annotations

import json
import random
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
