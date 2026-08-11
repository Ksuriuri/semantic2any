#!/usr/bin/env python3
"""Build the local duration-filtered s2mel training mirror from GCS shards."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import tarfile
import time
from typing import Any, Callable, TypeVar

import gcsfs


PROJECT = "noiz-430406"
SOURCE_PREFIX = "noiz-taiwan-audio-data/preprocessed"
DEFAULT_KEY_FILE = Path("/mnt/data_sdd/hhy/SpeechData/gcs-key.json")
DEFAULT_OUTPUT_ROOT = Path("/mnt/data_3t_1/datasets/preprocess/s2mel-train-data")
DEFAULT_DATASETS = (
    "ears",
    "expresso",
    "Genshin",
    "hi_fi_tts",
    "noiz-short",
    "StarRail",
    "vctk",
    "WutheringWaves",
)
DEFAULT_MIN_DURATION = 3.0
DEFAULT_MIN_SAMPLE_RATE = 0
DEFAULT_METADATA_WORKERS = 16
DEFAULT_WORKERS = 4
DEFAULT_MIN_SPEAKER_RECORDS = 2
DEFAULT_MAX_CER = 0.5
DEFAULT_ASR_PRIMARY = "cohere-transcribe-03-2026"
DEFAULT_ASR_SECONDARY = "granite-speech-4.1-2b-nar"
RESERVED_FREE_BYTES = 20 * 1024**3
COPY_CHUNK_BYTES = 8 * 1024**2
SYNC_STATE_FILE = ".sync-state.json"
DEFAULT_MAX_AUDIO_SECONDS = 30.0
DEFAULT_SEMANTIC_LOOKUP_NAME = "maskgct_lookup.pt"
SEMANTIC_CODEC_NAME = "maskgct"
SEMANTIC_CODEBOOKS = 1
CODE_LENGTH_DRIFT_TOLERANCE = 2.0
T = TypeVar("T")


def log(event: str, **fields: Any) -> None:
    print(
        json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


import re
import unicodedata

try:
    from rapidfuzz.distance import Levenshtein as _rf_lev
    _USE_RAPIDFUZZ = True
except ImportError:
    _USE_RAPIDFUZZ = False

_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3000, 0x303F),
    (0x3040, 0x309F), (0x30A0, 0x30FF), (0xAC00, 0xD7AF),
)

def _is_cjk_lang(language: str) -> bool:
    return language.lower() in ("zh", "ja", "ko", "cmn", "yue", "jpn", "kor", "chinese", "japanese", "korean")

def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.strip().lower())
    text = re.sub(r"[^\w]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

def _edit_distance(ref_seq: list | str, hyp_seq: list | str) -> int:
    if _USE_RAPIDFUZZ:
        if isinstance(ref_seq, list):
            return _rf_lev.distance(ref_seq, hyp_seq)
        return _rf_lev.distance(ref_seq, hyp_seq)
    n, m = len(ref_seq), len(hyp_seq)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if ref_seq[i - 1] == hyp_seq[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[m]

def compute_error_rate(reference: str, hypothesis: str, use_wer: bool = False) -> float:
    """Compute WER (word-level) or CER (char-level) after normalization."""
    ref_norm = _normalize_text(reference)
    hyp_norm = _normalize_text(hypothesis)
    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm:
        return 1.0
    if use_wer:
        ref_seq = ref_norm.split()
        hyp_seq = hyp_norm.split()
        n = len(ref_seq)
        if n == 0:
            return 1.0
        return _edit_distance(ref_seq, hyp_seq) / n
    else:
        n = len(ref_norm)
        if n == 0:
            return 1.0
        return _edit_distance(ref_norm, hyp_norm) / n


def _read_one_asr_shard(
    fs: gcsfs.GCSFileSystem,
    asr_path: str,
    attempts: int,
) -> dict[str, str]:
    texts: dict[str, str] = {}
    with retry(
        f"open {asr_path}",
        lambda p=asr_path: fs.open(p, "r"),
        attempts,
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("id")
            text = row.get("text")
            if rid and isinstance(text, str) and text.strip():
                texts[rid] = text.strip()
    return texts


def load_asr_texts(
    fs: gcsfs.GCSFileSystem,
    dataset_prefix: str,
    asr_model: str,
    attempts: int,
    workers: int = 16,
    keep_stems: set[str] | None = None,
) -> dict[str, str]:
    """Load ASR transcriptions concurrently into a dict keyed by record id."""
    asr_prefix = f"{dataset_prefix}/asr/{asr_model}"
    try:
        asr_paths = sorted(
            str(p) for p in retry(
                f"glob {asr_prefix}/*.jsonl",
                lambda: fs.glob(f"{asr_prefix}/*.jsonl"),
                attempts,
            )
        )
    except Exception:
        return {}
    if keep_stems is not None:
        # Only the shards we are going to scan; for podcast_en that is 1308 of
        # 22357 files per ASR model, and there are two models.
        asr_paths = [p for p in asr_paths if PurePosixPath(p).stem in keep_stems]
    if not asr_paths:
        return {}
    texts: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(asr_paths))) as executor:
        futures = {
            executor.submit(_read_one_asr_shard, fs, p, attempts): p
            for p in asr_paths
        }
        for future in as_completed(futures):
            asr_path = futures[future]
            try:
                shard_texts = future.result()
                texts.update(shard_texts)
            except Exception as exc:
                log("asr_load_warning", asr_path=asr_path, error=str(exc))
    return texts


def _read_one_maskgct_shard(
    fs: gcsfs.GCSFileSystem,
    jsonl_path: str,
    attempts: int,
) -> set[str]:
    ids: set[str] = set()
    with retry(
        f"open {jsonl_path}",
        lambda p=jsonl_path: fs.open(p, "r"),
        attempts,
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row.get("id")
            if rid:
                ids.add(rid)
    return ids


def list_code_shard_stems(
    fs: gcsfs.GCSFileSystem,
    dataset: str,
    attempts: int,
) -> set[str]:
    """Shard stems that already have maskGCT codes on GCS.

    Codes are encoded per shard, and a shard's codes jsonl, codes bin, metadata
    jsonl and audio tar all share one stem, so knowing which tars are worth
    scanning costs a single listing -- no shard contents are read.  Coverage is
    partial and grows over time (podcast_en: 1308 of 22357 shards on
    2026-08-11), and rows without codes are dropped by the maskGCT filter
    anyway, so scanning the rest is pure waste.
    """
    codes_prefix = f"{SOURCE_PREFIX}/{dataset}/features/maskGCT_codes"
    try:
        paths = retry(
            f"glob {codes_prefix}/*.jsonl",
            lambda: fs.glob(f"{codes_prefix}/*.jsonl"),
            attempts,
        )
    except Exception as exc:
        log("code_shards_list_failed", dataset=dataset, error=str(exc))
        return set()
    return {PurePosixPath(str(path)).stem for path in paths}


def restrict_to_code_shards(
    dataset: str,
    metadata_paths: list[str],
    audio_details: dict[str, dict[str, Any]],
    code_shard_stems: set[str],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Keep only the metadata shards and audio tars that have codes."""
    kept_metadata = [
        path for path in metadata_paths
        if PurePosixPath(path).stem in code_shard_stems
    ]
    kept_audio = {
        tar_key: details for tar_key, details in audio_details.items()
        if PurePosixPath(tar_key).stem in code_shard_stems
    }
    log(
        "code_shards_restriction",
        dataset=dataset,
        code_shards=len(code_shard_stems),
        metadata_shards_before=len(metadata_paths),
        metadata_shards_after=len(kept_metadata),
        audio_tars_before=len(audio_details),
        audio_tars_after=len(kept_audio),
    )
    if not kept_metadata or not kept_audio:
        raise FileNotFoundError(
            f"{dataset}: no metadata/audio shards overlap the maskGCT code shards"
        )
    return kept_metadata, kept_audio


def load_maskgct_code_ids(
    fs: gcsfs.GCSFileSystem,
    dataset_prefix: str,
    attempts: int,
    workers: int = 16,
) -> set[str]:
    """Load the set of record IDs that have maskGCT codes on GCS."""
    codes_prefix = f"{dataset_prefix}/features/maskGCT_codes"
    try:
        all_files = retry(
            f"glob {codes_prefix}/*.jsonl",
            lambda: fs.glob(f"{codes_prefix}/*.jsonl"),
            attempts,
        )
    except Exception:
        return set()
    jsonl_paths = sorted(str(p) for p in all_files)
    if not jsonl_paths:
        return set()
    ids: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(workers, len(jsonl_paths))) as executor:
        futures = {
            executor.submit(_read_one_maskgct_shard, fs, p, attempts): p
            for p in jsonl_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                shard_ids = future.result()
                ids.update(shard_ids)
            except Exception as exc:
                log("maskgct_ids_load_warning", path=path, error=str(exc))
    return ids


def retry(description: str, operation: Callable[[], T], attempts: int) -> T:
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts:
                raise
            wait_seconds = min(2 ** (attempt - 1), 15)
            log(
                "retry",
                description=description,
                attempt=attempt,
                attempts=attempts,
                wait_seconds=wait_seconds,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"{description} failed without raising")


def atomic_write_json(path: Path, value: Any) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    tmp_path.replace(path)


def _filter_params_key(args: argparse.Namespace) -> str:
    """Hash of filter parameters to detect config changes between runs."""
    params = {
        "min_duration": args.min_duration,
        "min_sample_rate": args.min_sample_rate,
        "max_cer": args.max_cer if not args.no_cer_filter else None,
        "min_speaker_records": args.min_speaker_records if not args.no_speaker_filter else None,
        "asr_primary": args.asr_primary if not args.no_cer_filter else None,
        "asr_secondary": args.asr_secondary if not args.no_cer_filter else None,
        "require_maskgct_codes": not args.no_require_maskgct_codes,
        "languages": sorted(args.languages.split(",")) if args.languages else None,
        "max_hours_per_lang": args.max_hours_per_lang if args.max_hours_per_lang > 0 else None,
        "codes_shards_only": args.codes_shards_only and not args.no_require_maskgct_codes,
    }
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def load_sync_state(output_root: Path) -> dict[str, Any]:
    state_path = output_root / SYNC_STATE_FILE
    if not state_path.is_file():
        return {}
    try:
        with state_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_sync_state(output_root: Path, state: dict[str, Any]) -> None:
    atomic_write_json(output_root / SYNC_STATE_FILE, state)


def detect_new_tars(
    fs: gcsfs.GCSFileSystem,
    dataset: str,
    attempts: int,
    sync_state: dict[str, Any],
    filter_key: str,
    code_shard_stems: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], bool]:
    """Compare GCS tar listing against saved state.

    Returns (current_audio_details, new_tar_keys, needs_rescan).
    needs_rescan is True if there are new tars or filter params changed.
    """
    dataset_prefix = f"{SOURCE_PREFIX}/{dataset}"
    audio_details = glob_details(fs, f"{dataset_prefix}/audio/*.tar", attempts)

    dataset_state = sync_state.get("datasets", {}).get(dataset, {})
    saved_filter_key = dataset_state.get("filter_key", "")
    saved_tars = set(dataset_state.get("synced_tars", []))

    current_tars = set(audio_details.keys())
    new_tars = sorted(current_tars - saved_tars)

    if saved_filter_key != filter_key:
        log(
            "incremental_filter_changed",
            dataset=dataset,
            reason="filter parameters changed since last sync",
        )
        return audio_details, list(current_tars), True

    if code_shard_stems is not None:
        # When the scan is restricted to coded shards, newly encoded codes -- not
        # newly uploaded tars -- are what makes a rescan worth doing.
        new_code_shards = code_shard_stems - set(
            dataset_state.get("synced_code_shards", [])
        )
        if new_code_shards:
            log(
                "incremental_new_code_shards",
                dataset=dataset,
                new_code_shards=len(new_code_shards),
                total_code_shards=len(code_shard_stems),
                examples=sorted(new_code_shards)[:5],
            )
            return audio_details, list(current_tars), True
        log(
            "incremental_no_new_code_shards",
            dataset=dataset,
            total_code_shards=len(code_shard_stems),
            new_uncoded_tars=len(new_tars),
        )
        return audio_details, [], False

    if new_tars:
        log(
            "incremental_new_tars",
            dataset=dataset,
            new_tar_count=len(new_tars),
            total_tars=len(current_tars),
            examples=new_tars[:5],
        )
        return audio_details, new_tars, True

    log("incremental_no_change", dataset=dataset, total_tars=len(current_tars))
    return audio_details, [], False


def parse_audio_path(dataset: str, audio_path: Any) -> tuple[str, str, str]:
    if not isinstance(audio_path, str) or not audio_path:
        raise ValueError(f"{dataset}: missing audio_path")
    path = PurePosixPath(audio_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{dataset}: unsafe audio_path: {audio_path}")

    tar_indexes = [
        index for index, part in enumerate(path.parts) if part.endswith(".tar")
    ]
    if len(tar_indexes) != 1:
        raise ValueError(
            f"{dataset}: expected exactly one tar segment in audio_path: {audio_path}"
        )
    tar_index = tar_indexes[0]
    tar_relative = PurePosixPath(*path.parts[: tar_index + 1]).as_posix()
    member = PurePosixPath(*path.parts[tar_index + 1 :]).as_posix()
    if not tar_relative.startswith("audio/") or not member:
        raise ValueError(f"{dataset}: invalid SpeechData audio_path: {audio_path}")

    member_basename = PurePosixPath(member).name
    if not member_basename.lower().endswith(".flac"):
        raise ValueError(f"{dataset}: expected a FLAC tar member: {audio_path}")
    return tar_relative, member, member_basename


def flat_output_name(
    tar_relative: str,
    member: str,
    member_basename: str,
    member_occurrence: int,
) -> str:
    """Disambiguate duplicate normalized tar member names deterministically."""
    shard_stem = PurePosixPath(tar_relative).stem
    extension = PurePosixPath(member_basename).suffix
    member_stem = member_basename[: -len(extension)] if extension else member_basename
    digest = hashlib.sha1(
        f"{tar_relative}/{member}#{member_occurrence}".encode("utf-8")
    ).hexdigest()[:12]
    candidate = f"{shard_stem}__{member_stem}__{digest}{extension}"
    if len(os.fsencode(candidate)) <= 240:
        return candidate
    shortened = member_stem[: 220 - len(shard_stem)]
    return f"{shard_stem}__{shortened}__{digest}{extension}"


def shard_partition(shard_stem: str, partitioned: bool) -> str:
    """Return the partition subdir (e.g. ``p030``) for a tar stem.

    Huge flat directories (millions of files with long common prefixes) can
    saturate ext4 htree hash buckets and fail with ENOSPC even when the disk
    has free space.  Partitioning into a few thousand-entry subdirectories
    avoids that failure mode entirely.
    """
    if not partitioned:
        return ""
    digits = "".join(ch for ch in shard_stem.rsplit("-", 1)[-1] if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot derive shard partition from {shard_stem!r}")
    return f"p{int(digits) // 1000:03d}"


@dataclass
class SelectedRecord:
    metadata: dict[str, Any]
    member: str
    member_occurrence: int
    basename: str
    expected_size: int | None = None
    subdir: str = ""


@dataclass
class DatasetPlan:
    name: str
    source_records: int = 0
    metadata_shards: int = 0
    selected_duration_seconds: float = 0.0
    referenced_tar_bytes: int = 0
    selected: list[SelectedRecord] = field(default_factory=list)
    by_tar: dict[str, dict[str, dict[int, SelectedRecord]]] = field(
        default_factory=dict
    )
    by_tar_basename: dict[str, dict[str, SelectedRecord]] = field(
        default_factory=dict
    )
    partitioned: bool = False


def glob_paths(
    fs: gcsfs.GCSFileSystem,
    pattern: str,
    attempts: int,
) -> list[str]:
    return sorted(
        str(path)
        for path in retry(
            f"glob {pattern}",
            lambda: fs.glob(pattern),
            attempts,
        )
    )


def glob_details(
    fs: gcsfs.GCSFileSystem,
    pattern: str,
    attempts: int,
) -> dict[str, dict[str, Any]]:
    result = retry(
        f"glob details {pattern}",
        lambda: fs.glob(pattern, detail=True),
        attempts,
    )
    if isinstance(result, dict):
        return {str(path): dict(info) for path, info in result.items()}
    return {
        str(path): retry(
            f"info {path}",
            lambda path=path: fs.info(str(path)),
            attempts,
        )
        for path in result
    }


def read_metadata_rows(
    fs: gcsfs.GCSFileSystem,
    metadata_path: str,
) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with fs.open(metadata_path, "rb") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{metadata_path}:{line_number}: metadata row is not an object"
                )
            rows.append((line_number, row))
    return rows


def iter_metadata_rows_bounded(
    fs: gcsfs.GCSFileSystem,
    metadata_paths: list[str],
    attempts: int,
    workers: int,
):
    """Read metadata concurrently with bounded memory and deterministic order."""
    path_iter = iter(metadata_paths)
    pending: deque[tuple[str, Future[list[tuple[int, dict[str, Any]]]]]] = deque()

    def read_with_retry(path: str) -> list[tuple[int, dict[str, Any]]]:
        return retry(
            f"read metadata {path}",
            lambda: read_metadata_rows(fs, path),
            attempts,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(workers * 2, len(metadata_paths))):
            path = next(path_iter, None)
            if path is None:
                break
            pending.append((path, executor.submit(read_with_retry, path)))

        while pending:
            metadata_path, future = pending.popleft()
            yield metadata_path, future.result()
            next_path = next(path_iter, None)
            if next_path is not None:
                pending.append(
                    (next_path, executor.submit(read_with_retry, next_path))
                )


def scan_dataset(
    fs: gcsfs.GCSFileSystem,
    dataset: str,
    min_duration: float,
    min_sample_rate: int,
    metadata_shard_sample: int,
    sample_seed: int,
    metadata_workers: int,
    attempts: int,
    *,
    max_cer: float = DEFAULT_MAX_CER,
    asr_primary: str = DEFAULT_ASR_PRIMARY,
    asr_secondary: str = DEFAULT_ASR_SECONDARY,
    skip_cer: bool = False,
    require_maskgct_codes: bool = False,
    languages: set[str] | None = None,
    max_hours_per_lang: float = 0,
    audio_details: dict[str, dict[str, Any]] | None = None,
    partitioned: bool = False,
    code_shard_stems: set[str] | None = None,
) -> DatasetPlan:
    dataset_prefix = f"{SOURCE_PREFIX}/{dataset}"
    metadata_paths = glob_paths(
        fs,
        f"{dataset_prefix}/metadata/*.jsonl",
        attempts,
    )
    if audio_details is None:
        audio_details = glob_details(
            fs,
            f"{dataset_prefix}/audio/*.tar",
            attempts,
        )
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata shards found for {dataset}")
    if not audio_details:
        raise FileNotFoundError(f"No audio tar shards found for {dataset}")

    if code_shard_stems:
        metadata_paths, audio_details = restrict_to_code_shards(
            dataset,
            metadata_paths,
            audio_details,
            code_shard_stems,
        )

    # Load ASR texts for CER filtering
    asr_primary_texts: dict[str, str] = {}
    asr_secondary_texts: dict[str, str] = {}
    if not skip_cer:
        asr_primary_used = asr_primary
        asr_secondary_used = asr_secondary
        asr_primary_texts = load_asr_texts(fs, dataset_prefix, asr_primary, attempts, keep_stems=code_shard_stems or None)
        asr_secondary_texts = load_asr_texts(fs, dataset_prefix, asr_secondary, attempts, keep_stems=code_shard_stems or None)
        # Auto-detect available ASR models if configured ones return empty
        if not asr_primary_texts or not asr_secondary_texts:
            try:
                asr_dirs = sorted(
                    p.rsplit("/", 1)[-1]
                    for p in retry(
                        f"ls {dataset_prefix}/asr",
                        lambda: fs.ls(f"{dataset_prefix}/asr"),
                        attempts,
                    )
                    if not p.endswith(".jsonl")
                )
            except Exception:
                asr_dirs = []
            if asr_dirs:
                if not asr_primary_texts and asr_dirs:
                    for candidate in asr_dirs:
                        if candidate != asr_secondary_used:
                            asr_primary_texts = load_asr_texts(fs, dataset_prefix, candidate, attempts, keep_stems=code_shard_stems or None)
                            if asr_primary_texts:
                                asr_primary_used = candidate
                                break
                if not asr_secondary_texts and len(asr_dirs) > 1:
                    for candidate in asr_dirs:
                        if candidate != asr_primary_used:
                            asr_secondary_texts = load_asr_texts(fs, dataset_prefix, candidate, attempts, keep_stems=code_shard_stems or None)
                            if asr_secondary_texts:
                                asr_secondary_used = candidate
                                break
        log(
            "asr_loaded",
            dataset=dataset,
            primary_model=asr_primary_used,
            primary_records=len(asr_primary_texts),
            secondary_model=asr_secondary_used,
            secondary_records=len(asr_secondary_texts),
        )
    # Load maskGCT code IDs if filtering is required
    maskgct_ids: set[str] = set()
    if require_maskgct_codes:
        maskgct_ids = load_maskgct_code_ids(fs, dataset_prefix, attempts, metadata_workers)
        log(
            "maskgct_ids_loaded",
            dataset=dataset,
            ids_count=len(maskgct_ids),
        )
        if not maskgct_ids:
            log("maskgct_ids_warning", dataset=dataset, reason="no maskGCT code IDs found, all records will be filtered out")

    source_metadata_shards = len(metadata_paths)
    if metadata_shard_sample and metadata_shard_sample < len(metadata_paths):
        metadata_paths = sorted(
            random.Random(f"{sample_seed}:{dataset}").sample(
                metadata_paths,
                metadata_shard_sample,
            )
        )

    plan = DatasetPlan(
        name=dataset,
        metadata_shards=len(metadata_paths),
        partitioned=partitioned,
    )
    seen_ids: dict[str, str] = {}
    seen_basenames: dict[str, str] = {}
    lang_hours: dict[str, float] = defaultdict(float)
    member_occurrences: dict[tuple[str, str], int] = defaultdict(int)

    all_langs_capped = False
    for metadata_path, rows in iter_metadata_rows_bounded(
        fs,
        metadata_paths,
        attempts,
        metadata_workers,
    ):
        if all_langs_capped:
            break
        for line_number, row in rows:
            plan.source_records += 1
            location = f"{metadata_path}:{line_number}"
            raw_duration = row.get("duration")
            if isinstance(raw_duration, bool) or not isinstance(
                raw_duration, (int, float)
            ):
                raise ValueError(f"{location}: duration is missing or non-numeric")
            duration = float(raw_duration)
            raw_sample_rate = row.get("sample_rate")
            if isinstance(raw_sample_rate, bool) or not isinstance(
                raw_sample_rate, (int, float)
            ):
                raise ValueError(f"{location}: sample_rate is missing or non-numeric")
            sample_rate = int(raw_sample_rate)

            tar_relative, member, member_basename = parse_audio_path(
                dataset,
                row.get("audio_path"),
            )
            tar_key = f"{dataset_prefix}/{tar_relative}"
            if tar_key not in audio_details:
                continue
            occurrence_key = (tar_key, member)
            member_occurrence = member_occurrences[occurrence_key]
            member_occurrences[occurrence_key] += 1

            if duration <= min_duration or sample_rate < min_sample_rate:
                continue

            # Cheap filters first: language, hour cap, dedup, maskGCT
            record_language = row.get("language", "").lower()
            if languages and record_language not in languages:
                continue
            if max_hours_per_lang > 0 and lang_hours[record_language] >= max_hours_per_lang:
                continue

            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{location}: missing id")
            if record_id in seen_ids:
                continue

            if require_maskgct_codes and record_id not in maskgct_ids:
                continue

            # WER/CER filtering (WER for English-like, CER for CJK)
            if not skip_cer:
                record_lang = row.get("language", dataset.split("_")[-1] if "single_speaker_" in dataset else "")
                use_wer = not _is_cjk_lang(record_lang)
                metadata_text = row.get("text")
                has_text = isinstance(metadata_text, str) and metadata_text.strip()
                if has_text:
                    asr_text = asr_primary_texts.get(record_id)
                    if asr_text:
                        err = compute_error_rate(metadata_text.strip(), asr_text, use_wer=use_wer)
                        if err > max_cer:
                            continue
                else:
                    primary_text = asr_primary_texts.get(record_id)
                    secondary_text = asr_secondary_texts.get(record_id)
                    if primary_text and secondary_text:
                        err = compute_error_rate(primary_text, secondary_text, use_wer=use_wer)
                        if err > max_cer:
                            continue
                    elif not primary_text and not secondary_text:
                        continue

            basename = flat_output_name(
                tar_relative,
                member,
                member_basename,
                member_occurrence,
            )
            if basename in seen_basenames:
                raise ValueError(
                    f"{dataset}: flat filename collision {basename!r}: "
                    f"{seen_basenames[basename]} and {location}"
                )

            output_metadata = dict(row)
            output_metadata["dataset"] = dataset
            subdir = shard_partition(PurePosixPath(tar_relative).stem, partitioned)
            if subdir:
                output_metadata["audio_path"] = (
                    f"../audios/{dataset}/{subdir}/{basename}"
                )
            else:
                output_metadata["audio_path"] = f"../audios/{dataset}/{basename}"
            selected = SelectedRecord(
                metadata=output_metadata,
                member=member,
                member_occurrence=member_occurrence,
                basename=basename,
                subdir=subdir,
            )
            records_by_occurrence = plan.by_tar.setdefault(tar_key, {}).setdefault(
                member,
                {},
            )
            if member_occurrence in records_by_occurrence:
                raise ValueError(
                    f"{dataset}: duplicate tar member occurrence "
                    f"{member!r}#{member_occurrence}"
                )

            records_by_occurrence[member_occurrence] = selected
            plan.selected.append(selected)
            plan.selected_duration_seconds += duration
            lang_hours[record_language] += duration / 3600.0
            if max_hours_per_lang > 0 and languages and all(
                lang_hours.get(lg, 0) >= max_hours_per_lang for lg in languages
            ):
                all_langs_capped = True
                break
            seen_ids[record_id] = location
            seen_basenames[basename] = location

    for tar_key in plan.by_tar:
        raw_size = audio_details[tar_key].get("size")
        if not isinstance(raw_size, (int, float)) or int(raw_size) <= 0:
            raise ValueError(f"Missing positive object size for gs://{tar_key}")
        plan.referenced_tar_bytes += int(raw_size)

    log(
        "preflight_dataset",
        dataset=dataset,
        metadata_shards=plan.metadata_shards,
        source_metadata_shards=source_metadata_shards,
        sampled_metadata_shards=bool(metadata_shard_sample),
        metadata_workers=metadata_workers,
        source_records=plan.source_records,
        selected_records=len(plan.selected),
        selected_duration_hours=round(plan.selected_duration_seconds / 3600.0, 3),
        min_sample_rate=min_sample_rate,
        referenced_tars=len(plan.by_tar),
        referenced_tar_bytes=plan.referenced_tar_bytes,
    )
    return plan


def write_metadata_tmp(output_root: Path, plan: DatasetPlan) -> Path:
    (output_root / "audios" / plan.name).mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_tmp = metadata_dir / f"{plan.name}.jsonl.tmp"
    with metadata_tmp.open("w", encoding="utf-8") as file_obj:
        for selected in plan.selected:
            json.dump(
                selected.metadata,
                file_obj,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    return metadata_tmp


def valid_flac(path: Path, expected_size: int | None = None) -> bool:
    if not path.is_file():
        return False
    size = path.stat().st_size
    if size <= 0 or (expected_size is not None and size != expected_size):
        return False
    with path.open("rb") as file_obj:
        return file_obj.read(4) == b"fLaC"


def copy_tar_member(
    tar_file: tarfile.TarFile,
    member_info: tarfile.TarInfo,
    destination: Path,
    tmp_suffix: str = ".part",
) -> None:
    source = tar_file.extractfile(member_info)
    if source is None:
        raise FileNotFoundError(f"Could not extract tar member {member_info.name}")
    tmp_path = destination.with_name(f".{destination.name}{tmp_suffix}")
    tmp_path.unlink(missing_ok=True)
    try:
        with source, tmp_path.open("wb") as output:
            shutil.copyfileobj(source, output, COPY_CHUNK_BYTES)
            output.flush()
            os.fsync(output.fileno())
        if tmp_path.stat().st_size != member_info.size:
            raise IOError(
                f"Size mismatch for {member_info.name}: "
                f"expected={member_info.size} actual={tmp_path.stat().st_size}"
            )
        tmp_path.replace(destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def extract_tar_once(
    fs: gcsfs.GCSFileSystem,
    tar_key: str,
    selected_by_member: dict[str, dict[int, SelectedRecord]],
    dataset_dir: Path,
    subdir: str,
) -> tuple[int, int]:
    remaining = {
        (member, occurrence)
        for member, records_by_occurrence in selected_by_member.items()
        for occurrence in records_by_occurrence
    }
    occurrences: dict[str, int] = defaultdict(int)
    extracted = 0
    reused = 0

    with fs.open(tar_key, "rb") as raw_file:
        with tarfile.open(fileobj=raw_file, mode="r|*") as tar_file:
            for member_info in tar_file:
                occurrence = occurrences[member_info.name]
                occurrences[member_info.name] += 1
                member_key = (member_info.name, occurrence)
                if member_key not in remaining:
                    continue
                if not member_info.isfile():
                    raise ValueError(
                        f"Selected tar member is not a file: "
                        f"{tar_key}/{member_info.name}#{occurrence}"
                    )

                selected = selected_by_member[member_info.name][occurrence]
                selected.expected_size = int(member_info.size)
                destination = dataset_dir / subdir / selected.basename
                if valid_flac(destination, member_info.size):
                    reused += 1
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    copy_tar_member(tar_file, member_info, destination)
                    extracted += 1
                remaining.remove(member_key)
                if not remaining:
                    break

    if remaining:
        raise FileNotFoundError(
            f"Missing {len(remaining)} selected members in gs://{tar_key}; "
            f"examples={sorted(remaining)[:5]}"
        )
    return extracted, reused


def reuse_complete_shard(
    selected_by_member: dict[str, dict[int, SelectedRecord]],
    dataset_dir: Path,
    subdir: str,
) -> int | None:
    sizes: list[tuple[SelectedRecord, int]] = []
    for records_by_occurrence in selected_by_member.values():
        for selected in records_by_occurrence.values():
            path = dataset_dir / subdir / selected.basename
            if not valid_flac(path):
                return None
            sizes.append((selected, path.stat().st_size))
    for selected, size in sizes:
        selected.expected_size = size
    return len(sizes)


def extract_dataset(
    fs: gcsfs.GCSFileSystem,
    output_root: Path,
    plan: DatasetPlan,
    attempts: int,
    workers: int,
) -> None:
    dataset_dir = output_root / "audios" / plan.name
    tar_items = sorted(plan.by_tar.items())
    extracted_total = 0
    reused_total = 0
    pending: list[
        tuple[int, str, dict[str, dict[int, SelectedRecord]]]
    ] = []

    for shard_index, (tar_key, selected_by_member) in enumerate(tar_items, start=1):
        selected_count = sum(
            len(records) for records in selected_by_member.values()
        )
        subdir = shard_partition(PurePosixPath(tar_key).stem, plan.partitioned)
        reused = reuse_complete_shard(selected_by_member, dataset_dir, subdir)
        if reused is not None:
            reused_total += reused
            log(
                "extract_shard",
                dataset=plan.name,
                shard=PurePosixPath(tar_key).name,
                shard_index=shard_index,
                shard_total=len(tar_items),
                selected_members=selected_count,
                extracted=0,
                reused=reused,
                source_skipped=True,
            )
        else:
            pending.append((shard_index, tar_key, selected_by_member))

    def extract_with_retry(
        tar_key: str,
        selected_by_member: dict[str, dict[int, SelectedRecord]],
        subdir: str,
    ) -> tuple[int, int]:
        return retry(
            f"extract gs://{tar_key}",
            lambda: extract_tar_once(
                fs,
                tar_key,
                selected_by_member,
                dataset_dir,
                subdir,
            ),
            attempts,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[
            Future[tuple[int, int]],
            tuple[int, str, dict[str, dict[int, SelectedRecord]]],
        ] = {
            executor.submit(
                extract_with_retry,
                tar_key,
                selected_by_member,
                shard_partition(PurePosixPath(tar_key).stem, plan.partitioned),
            ): (shard_index, tar_key, selected_by_member)
            for shard_index, tar_key, selected_by_member in pending
        }
        try:
            for future in as_completed(futures):
                shard_index, tar_key, selected_by_member = futures[future]
                extracted, reused = future.result()
                extracted_total += extracted
                reused_total += reused
                log(
                    "extract_shard",
                    dataset=plan.name,
                    shard=PurePosixPath(tar_key).name,
                    shard_index=shard_index,
                    shard_total=len(tar_items),
                    selected_members=sum(
                        len(records) for records in selected_by_member.values()
                    ),
                    extracted=extracted,
                    reused=reused,
                    source_skipped=False,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    log(
        "extract_dataset_complete",
        dataset=plan.name,
        selected_records=len(plan.selected),
        extracted=extracted_total,
        reused=reused_total,
        workers=workers,
    )


def build_plan_from_metadata_tmp(
    output_root: Path,
    dataset: str,
    partitioned: bool,
    fs: gcsfs.GCSFileSystem,
    attempts: int,
) -> tuple[DatasetPlan, dict[str, dict[str, Any]]]:
    """Rebuild a DatasetPlan from a previous run's metadata tmp file.

    This avoids re-scanning and re-CER-filtering millions of source rows after
    a crash: the tmp file already contains the exact selected record set.
    """
    metadata_tmp = output_root / "metadata" / f"{dataset}.jsonl.tmp"
    if not metadata_tmp.is_file():
        raise FileNotFoundError(
            f"resume requires {metadata_tmp} (a prior scan's metadata tmp)"
        )
    dataset_prefix = f"{SOURCE_PREFIX}/{dataset}"
    audio_details = glob_details(fs, f"{dataset_prefix}/audio/*.tar", attempts)
    plan = DatasetPlan(
        name=dataset,
        metadata_shards=0,
        partitioned=partitioned,
    )
    with metadata_tmp.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            old_audio_path = row.get("audio_path")
            if not isinstance(old_audio_path, str) or not old_audio_path:
                raise ValueError(
                    f"{metadata_tmp}:{line_number}: missing audio_path"
                )
            basename = PurePosixPath(old_audio_path).name
            shard_stem = basename.split("__", 1)[0]
            tar_relative = f"audio/{shard_stem}.tar"
            tar_key = f"{dataset_prefix}/{tar_relative}"
            if tar_key not in audio_details:
                raise ValueError(
                    f"{metadata_tmp}:{line_number}: tar not on GCS: {tar_key}"
                )
            subdir = shard_partition(shard_stem, partitioned)
            if subdir:
                row["audio_path"] = (
                    f"../audios/{dataset}/{subdir}/{basename}"
                )
            else:
                row["audio_path"] = f"../audios/{dataset}/{basename}"
            selected = SelectedRecord(
                metadata=row,
                member="",
                member_occurrence=0,
                basename=basename,
                subdir=subdir,
            )
            plan.selected.append(selected)
            plan.by_tar.setdefault(tar_key, {})
            plan.by_tar_basename.setdefault(tar_key, {})[basename] = selected
            plan.selected_duration_seconds += float(row.get("duration", 0))
    plan.referenced_tar_bytes = sum(
        int(audio_details[tar_key].get("size", 0))
        for tar_key in plan.by_tar_basename
    )
    plan.source_records = len(plan.selected)
    referenced_audio_details = {
        tar_key: audio_details[tar_key]
        for tar_key in plan.by_tar_basename
    }
    log(
        "resume_plan_loaded",
        dataset=dataset,
        selected_records=len(plan.selected),
        tars=len(plan.by_tar_basename),
        referenced_tar_bytes=plan.referenced_tar_bytes,
        partitioned=partitioned,
    )
    return plan, referenced_audio_details


def extract_dataset_resume(
    fs: gcsfs.GCSFileSystem,
    output_root: Path,
    plan: DatasetPlan,
    attempts: int,
    workers: int,
) -> None:
    """Resume extraction using a plan rebuilt from metadata tmp.

    Completed shards are detected by the presence of every expected FLAC in
    its partition subdir; incomplete shards are re-indexed from GCS by
    recomputing each tar member's flat output name.
    """
    import threading

    dataset_dir = output_root / "audios" / plan.name
    validated_sidecar = dataset_dir / ".resume-validated.json"
    validated_lock = threading.Lock()
    validated: set[str] = set()
    if validated_sidecar.is_file():
        try:
            validated = set(json.loads(validated_sidecar.read_text()))
        except (json.JSONDecodeError, OSError):
            validated = set()
    log(
        "resume_validated_loaded",
        dataset=plan.name,
        validated_tars=len(validated),
    )

    def mark_validated(shard_stem: str) -> None:
        with validated_lock:
            validated.add(shard_stem)
            tmp = validated_sidecar.with_name(
                f"{validated_sidecar.name}.tmp"
            )
            tmp.write_text(
                json.dumps(sorted(validated), ensure_ascii=False)
            )
            tmp.replace(validated_sidecar)

    tar_items = sorted(plan.by_tar_basename.items())
    extracted_total = 0
    reused_total = 0
    pending: list[tuple[int, str, dict[str, SelectedRecord]]] = []

    def check_shard(
        args: tuple[int, tuple[str, dict[str, SelectedRecord]]],
    ) -> tuple[int, str, dict[str, SelectedRecord], int | None]:
        shard_index, (tar_key, needed) = args
        shard_stem = PurePosixPath(tar_key).stem
        subdir = shard_partition(shard_stem, plan.partitioned)
        if shard_stem in validated:
            try:
                for selected in needed.values():
                    selected.expected_size = (
                        dataset_dir / subdir / selected.basename
                    ).stat().st_size
                return shard_index, tar_key, needed, len(needed)
            except FileNotFoundError:
                return shard_index, tar_key, needed, None
        reused = 0
        for selected in needed.values():
            path = dataset_dir / subdir / selected.basename
            if not valid_flac(path):
                return shard_index, tar_key, needed, None
            selected.expected_size = path.stat().st_size
            reused += 1
        mark_validated(shard_stem)
        return shard_index, tar_key, needed, reused

    with ThreadPoolExecutor(max_workers=max(2, workers * 2)) as executor:
        reuse_futures = {
            executor.submit(check_shard, item): item
            for item in enumerate(tar_items, start=1)
        }
        for future in as_completed(reuse_futures):
            shard_index, tar_key, needed, reused = future.result()
            if reused is None:
                pending.append((shard_index, tar_key, needed))
                continue
            reused_total += reused
            log(
                "extract_shard",
                dataset=plan.name,
                shard=PurePosixPath(tar_key).name,
                shard_index=shard_index,
                shard_total=len(tar_items),
                selected_members=len(needed),
                extracted=0,
                reused=reused,
                source_skipped=True,
            )

    if pending:
        # GCS from R0905 can be as slow as ~0.3-0.5 MB/s (444MB tar ≈ 15-25
        # min); use a generous timeout so slow-but-alive streams finish while
        # still catching true connection hangs.
        MAX_TAR_SECONDS = 2400

        def process_tar(
            tar_key: str,
            needed: dict[str, SelectedRecord],
            tmp_suffix: str,
        ) -> tuple[int, int]:
            tar_relative = f"audio/{PurePosixPath(tar_key).name}"
            subdir = shard_partition(PurePosixPath(tar_key).stem, plan.partitioned)
            occurrences: dict[str, int] = defaultdict(int)
            remaining = set(needed)
            extracted = 0
            reused = 0

            def extract_with_matching() -> None:
                nonlocal extracted, reused
                with retry(
                    f"extract gs://{tar_key}",
                    lambda: fs.open(tar_key, "rb"),
                    attempts,
                ) as raw_file:
                    with tarfile.open(fileobj=raw_file, mode="r|*") as tar_file:
                        for member_info in tar_file:
                            occurrence = occurrences[member_info.name]
                            occurrences[member_info.name] += 1
                            if not member_info.isfile():
                                continue
                            member_basename = PurePosixPath(member_info.name).name
                            candidate = flat_output_name(
                                tar_relative,
                                member_info.name,
                                member_basename,
                                occurrence,
                            )
                            selected = needed.get(candidate)
                            if selected is None:
                                continue
                            destination = dataset_dir / subdir / candidate
                            if valid_flac(destination, member_info.size):
                                selected.expected_size = int(member_info.size)
                                reused += 1
                            else:
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                copy_tar_member(
                                    tar_file,
                                    member_info,
                                    destination,
                                    tmp_suffix,
                                )
                                selected.expected_size = int(member_info.size)
                                extracted += 1
                            remaining.discard(candidate)

            retry(f"extract gs://{tar_key}", extract_with_matching, attempts)
            if remaining:
                raise FileNotFoundError(
                    f"resume: {len(remaining)} selected members missing from "
                    f"gs://{tar_key}: {sorted(remaining)[:5]}"
                )
            mark_validated(PurePosixPath(tar_key).stem)
            return extracted, reused

        still_pending = list(pending)
        for attempt in range(1, attempts + 1):
            if not still_pending:
                break
            executor = ThreadPoolExecutor(max_workers=workers)
            futures = {
                executor.submit(
                    process_tar,
                    tar_key,
                    needed,
                    f".part.{os.getpid()}.{attempt}.{threading.get_ident()}",
                ): (shard_index, tar_key, needed)
                for shard_index, tar_key, needed in still_pending
            }
            try:
                done, not_done = wait(futures, timeout=MAX_TAR_SECONDS)
                for future in done:
                    shard_index, tar_key, needed = futures[future]
                    extracted, reused = future.result()
                    extracted_total += extracted
                    reused_total += reused
                    log(
                        "extract_shard",
                        dataset=plan.name,
                        shard=PurePosixPath(tar_key).name,
                        shard_index=shard_index,
                        shard_total=len(tar_items),
                        selected_members=len(needed),
                        extracted=extracted,
                        reused=reused,
                        source_skipped=False,
                    )
                if not_done:
                    log(
                        "extract_shard_timeout",
                        attempt=attempt,
                            tars=len(not_done),
                            timeout_seconds=MAX_TAR_SECONDS,
                            examples=[
                                PurePosixPath(futures[f][1]).name
                                for f in list(not_done)[:5]
                            ],
                        )
                    still_pending = [futures[f] for f in not_done]
                    for future in not_done:
                        future.cancel()
                else:
                    still_pending = []
            except BaseException:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            executor.shutdown(wait=False, cancel_futures=True)
        if still_pending:
            raise TimeoutError(
                f"resume: {len(still_pending)} tars still pending after "
                f"{attempts} attempts"
            )

    log(
        "extract_dataset_resume_complete",
        dataset=plan.name,
        selected_records=len(plan.selected),
        extracted=extracted_total,
        reused=reused_total,
        workers=workers,
    )


def verify_and_finalize_dataset(
    output_root: Path,
    plan: DatasetPlan,
    metadata_tmp: Path,
    min_duration: float,
    min_sample_rate: int,
) -> dict[str, Any]:
    dataset_dir = output_root / "audios" / plan.name
    if plan.partitioned:
        expected_names = {
            f"{selected.subdir}/{selected.basename}" for selected in plan.selected
        }
        actual_names = {
            f"{path.parent.name}/{path.name}"
            for path in dataset_dir.glob("*/*.flac")
            if path.is_file()
        }
    else:
        expected_names = {selected.basename for selected in plan.selected}
        actual_names = {
            path.name for path in dataset_dir.glob("*.flac") if path.is_file()
        }
    missing = expected_names - actual_names
    if missing:
        raise ValueError(
            f"{plan.name}: missing expected FLACs: "
            f"{sorted(missing)[:5]} ({len(missing)} total)"
        )
    extras = actual_names - expected_names
    if extras:
        log(
            "verify_extras",
            dataset=plan.name,
            extra_count=len(extras),
            note="leftover files from previous sync, safe to ignore",
        )

    total_bytes = 0
    for selected in plan.selected:
        path = dataset_dir / selected.subdir / selected.basename
        if selected.expected_size is None or not valid_flac(
            path, selected.expected_size
        ):
            raise ValueError(f"{plan.name}: invalid output FLAC: {path}")
        total_bytes += path.stat().st_size

    expected_by_path = {
        (
            f"../audios/{plan.name}/{selected.subdir}/{selected.basename}"
            if selected.subdir
            else f"../audios/{plan.name}/{selected.basename}"
        ): (dataset_dir / selected.subdir / selected.basename)
        for selected in plan.selected
    }
    expected_audio_paths = set(expected_by_path)
    metadata_audio_paths: set[str] = set()
    rows = 0
    with metadata_tmp.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            row = json.loads(line)
            duration = row.get("duration")
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise ValueError(f"{metadata_tmp}:{line_number}: invalid duration")
            if float(duration) <= min_duration:
                raise ValueError(
                    f"{metadata_tmp}:{line_number}: duration does not pass filter"
                )
            sample_rate = row.get("sample_rate")
            if isinstance(sample_rate, bool) or not isinstance(
                sample_rate, (int, float)
            ):
                raise ValueError(f"{metadata_tmp}:{line_number}: invalid sample_rate")
            if int(sample_rate) < min_sample_rate:
                raise ValueError(
                    f"{metadata_tmp}:{line_number}: sample_rate does not pass filter"
                )
            audio_path = row.get("audio_path")
            if not isinstance(audio_path, str) or audio_path not in expected_audio_paths:
                raise ValueError(
                    f"{metadata_tmp}:{line_number}: invalid audio_path {audio_path!r}"
                )
            if audio_path in metadata_audio_paths:
                raise ValueError(f"{metadata_tmp}:{line_number}: duplicate audio_path")
            resolved = (metadata_tmp.parent / audio_path).resolve()
            expected = expected_by_path[audio_path].resolve()
            if resolved != expected:
                raise ValueError(
                    f"{metadata_tmp}:{line_number}: audio_path escapes dataset"
                )
            metadata_audio_paths.add(audio_path)
            rows += 1

    if rows != len(plan.selected) or metadata_audio_paths != expected_audio_paths:
        raise ValueError(
            f"{plan.name}: metadata mismatch: rows={rows}, "
            f"selected={len(plan.selected)}"
        )

    metadata_path = metadata_tmp.parent / f"{plan.name}.jsonl"
    metadata_tmp.replace(metadata_path)
    (dataset_dir / "metadata.jsonl").unlink(missing_ok=True)
    (dataset_dir / "metadata.jsonl.tmp").unlink(missing_ok=True)
    result = {
        "dataset": plan.name,
        "metadata_path": str(metadata_path),
        "metadata_shards": plan.metadata_shards,
        "source_records": plan.source_records,
        "selected_records": len(plan.selected),
        "selected_duration_seconds": plan.selected_duration_seconds,
        "selected_duration_hours": plan.selected_duration_seconds / 3600.0,
        "min_duration_exclusive": min_duration,
        "min_sample_rate_inclusive": min_sample_rate,
        "referenced_tars": len(plan.by_tar),
        "referenced_tar_bytes": plan.referenced_tar_bytes,
        "output_bytes": total_bytes,
    }
    log("verify_dataset_complete", **result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download selected SpeechData FLACs from GCS into flat per-dataset "
            "directories and write one centralized JSONL per dataset."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION)
    parser.add_argument(
        "--min-sample-rate",
        type=int,
        default=DEFAULT_MIN_SAMPLE_RATE,
        help="Keep rows whose sample_rate is at least this value.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--metadata-workers",
        type=int,
        default=DEFAULT_METADATA_WORKERS,
        help="Concurrent metadata shard readers with bounded in-flight work.",
    )
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument(
        "--sample-metadata",
        type=int,
        default=0,
        help="Print a deterministic reservoir sample of selected metadata rows.",
    )
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument(
        "--metadata-shard-sample",
        type=int,
        default=0,
        help="Randomly scan only this many metadata shards; use with preflight-only.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("download_summary.json"),
        help="Summary filename written under output-root.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset child prefixes under gs://.../preprocessed/.",
    )
    parser.add_argument(
        "--min-speaker-records",
        type=int,
        default=DEFAULT_MIN_SPEAKER_RECORDS,
        help="Drop speakers with fewer than N records after duration filtering.",
    )
    parser.add_argument(
        "--max-cer",
        type=float,
        default=DEFAULT_MAX_CER,
        help="Max CER between text and ASR transcription (drop if CER > this).",
    )
    parser.add_argument(
        "--asr-primary",
        default=DEFAULT_ASR_PRIMARY,
        help="Primary ASR model folder name for CER computation.",
    )
    parser.add_argument(
        "--asr-secondary",
        default=DEFAULT_ASR_SECONDARY,
        help="Secondary ASR model folder for cross-ASR CER when text is null.",
    )
    parser.add_argument("--no-cer-filter", action="store_true", help="Skip CER filtering.")
    parser.add_argument("--no-speaker-filter", action="store_true", help="Skip speaker count filtering.")
    parser.add_argument("--no-require-maskgct-codes", action="store_true",
                        help="Disable filtering records by maskGCT codes availability.")
    parser.add_argument("--languages", type=str, default="",
                        help="Comma-separated language codes to keep (e.g. en,zh,de). Empty = all.")
    parser.add_argument("--max-hours-per-lang", type=float, default=0,
                        help="Approximate max hours to keep per language. 0 = unlimited.")
    parser.add_argument("--no-pull-maskgct-codes", action="store_true",
                        help="Skip downloading maskGCT semantic codes from GCS.")
    parser.add_argument("--no-codes-shards-only", dest="codes_shards_only",
                        action="store_false",
                        help=(
                            "Scan every metadata shard instead of only the "
                            "shards that already have maskGCT codes. Codes "
                            "cover a small fraction of the shards, so this is "
                            "much slower for no extra rows."
                        ))
    parser.add_argument("--semantic-lookup", type=Path, default=None,
                        help=(
                            "MaskGCT code->embedding table stamped into the "
                            "manifests. Default: "
                            f"<output-root>/maskgct-codes/{DEFAULT_SEMANTIC_LOOKUP_NAME}"
                        ))
    parser.add_argument("--max-audio-seconds", type=float,
                        default=DEFAULT_MAX_AUDIO_SECONDS,
                        help=(
                            "Drop coded rows longer than this, matching the "
                            "training config; codes cover the whole file while "
                            "training truncates the audio. 0 disables."
                        ))
    parser.add_argument("--force-rescan", action="store_true",
                        help="Ignore sync state and re-scan all datasets from scratch.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--partitioned-datasets",
        type=str,
        default="",
        help=(
            "Comma-separated dataset names whose audio is written into "
            "partition subdirectories (avoids ext4 htree ENOSPC in huge "
            "flat directories)."
        ),
    )
    parser.add_argument(
        "--resume-dataset",
        type=str,
        default="",
        help=(
            "Resume extraction/codes for one dataset from its existing "
            "metadata/<dataset>.jsonl.tmp instead of re-scanning GCS."
        ),
    )
    return parser.parse_args()



def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_semantic_lookup(
    output_root: Path,
    lookup_path: Path | None,
) -> tuple[Path, str]:
    """Locate the shared MaskGCT code->embedding table and hash it.

    Training refuses a batch whose rows disagree on either the path or the
    checksum, so one canonical file per output root is the whole contract.  It
    is derived from the frozen RepCodec weights (`quantizer.vq2emb`), which is
    why the sync only consumes it and never builds it.
    """
    resolved = (
        lookup_path
        if lookup_path is not None
        else output_root / "maskgct-codes" / DEFAULT_SEMANTIC_LOOKUP_NAME
    ).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"maskGCT lookup table not found: {resolved}. Build it from the "
            "RepCodec weights (see scripts/maskgct_codes_io/README.md) or pass "
            "--semantic-lookup; without it the pulled codes are unusable and "
            "training silently re-encodes audio instead."
        )
    return resolved.resolve(), sha256_file(resolved)


def unusable_code_row(row: dict[str, Any], max_audio_seconds: float) -> str | None:
    """Why this coded row cannot be trained on, or None if it can.

    Both rules exist because the codes cover the whole file while training
    truncates audio at `max_audio_seconds`:
    - `too_long`: the prompt slice could reference codes the mel never has.
    - `code_drift`: the stored code count disagrees with duration * fps, so the
      row was encoded from different audio than the metadata describes (worst
      case seen in v2: a 61.8 s podcast row carrying 303 codes = 6.1 s).
    """
    duration = row.get("duration")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        return "code_drift"
    duration = float(duration)
    if max_audio_seconds > 0 and duration > max_audio_seconds:
        return "too_long"
    code_length = row.get("semantic_code_length")
    fps = row.get("semantic_frame_rate")
    if not isinstance(code_length, int) or not isinstance(fps, (int, float)):
        return "code_drift"
    if abs(code_length - duration * float(fps)) > CODE_LENGTH_DRIFT_TOLERANCE:
        return "code_drift"
    return None


def download_maskgct_codes(
    fs: gcsfs.GCSFileSystem,
    output_root: Path,
    plans: list,
    attempts: int,
    workers: int,
    *,
    semantic_lookup_path: Path,
    semantic_lookup_sha256: str,
    max_audio_seconds: float,
) -> None:
    """Download maskGCT semantic codes and write training-ready manifests."""
    codes_root = output_root / "maskgct-codes"
    codes_root.mkdir(parents=True, exist_ok=True)
    bins_dir = codes_root / "bins"
    bins_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir = codes_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    for plan in plans:
        dataset = plan.name
        selected_ids = {s.metadata.get("id") for s in plan.selected}
        if not selected_ids:
            continue

        dataset_prefix = f"{SOURCE_PREFIX}/{dataset}"
        codes_prefix = f"{dataset_prefix}/features/maskGCT_codes"

        # List available shards
        try:
            all_files = retry(
                f"ls {codes_prefix}",
                lambda: fs.ls(codes_prefix),
                attempts,
            )
        except Exception as exc:
            log("maskgct_skip", dataset=dataset, reason=str(exc))
            continue

        bin_files = sorted(f for f in all_files if f.endswith(".u2.bin"))
        jsonl_files = sorted(f for f in all_files if f.endswith(".jsonl"))

        if not bin_files or not jsonl_files:
            log("maskgct_skip", dataset=dataset, reason="no bin/jsonl files")
            continue

        # Scan JSONL to find which shards contain selected records
        needed_bins: set[str] = set()
        filtered_records: list[str] = []
        total_scanned = 0
        dropped_too_long = 0
        dropped_code_drift = 0

        for jsonl_path in jsonl_files:
            try:
                with retry(
                    f"open {jsonl_path}",
                    lambda p=jsonl_path: fs.open(p, "r"),
                    attempts,
                ) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        total_scanned += 1
                        row = json.loads(line)
                        record_id = row.get("id")
                        if record_id in selected_ids:
                            reason = unusable_code_row(row, max_audio_seconds)
                            if reason == "too_long":
                                dropped_too_long += 1
                                continue
                            if reason == "code_drift":
                                dropped_code_drift += 1
                                continue
                            code_path = row.get("semantic_code_path", "")
                            if code_path:
                                needed_bins.add(code_path)
                            filtered_records.append(line)
            except Exception as exc:
                log("maskgct_jsonl_error", dataset=dataset, path=jsonl_path, error=str(exc))

        log(
            "maskgct_scan",
            dataset=dataset,
            total_scanned=total_scanned,
            filtered_records=len(filtered_records),
            needed_bins=len(needed_bins),
            dropped_too_long=dropped_too_long,
            dropped_code_drift=dropped_code_drift,
        )

        # Download needed bin shards
        dataset_bins_dir = bins_dir / dataset
        dataset_bins_dir.mkdir(parents=True, exist_ok=True)

        needed_bin_files = [
            bin_file for bin_file in bin_files
            if bin_file.rsplit("/", 1)[-1] in needed_bins
        ]
        bin_log_lock = __import__("threading").Lock()

        def download_one_bin(bin_file: str) -> None:
            bin_name = bin_file.rsplit("/", 1)[-1]
            target = dataset_bins_dir / bin_name
            if target.is_file() and target.stat().st_size > 0:
                return
            tmp = target.with_name(f".{target.name}.tmp")
            try:
                with retry(
                    f"download {bin_file}",
                    lambda p=bin_file: fs.open(p, "rb"),
                    attempts,
                ) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp.replace(target)
                with bin_log_lock:
                    log(
                        "maskgct_bin_downloaded",
                        dataset=dataset,
                        file=bin_name,
                        size_mb=round(target.stat().st_size / 1024 / 1024, 1),
                    )
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                with bin_log_lock:
                    log(
                        "maskgct_bin_error",
                        dataset=dataset,
                        file=bin_name,
                        error=str(exc),
                    )

        with ThreadPoolExecutor(max_workers=max(2, workers)) as executor:
            futures = [
                executor.submit(download_one_bin, bin_file)
                for bin_file in needed_bin_files
            ]
            for future in as_completed(futures):
                future.result()

        # Write filtered manifest
        manifest_path = manifests_dir / f"{dataset}.jsonl"
        manifest_tmp = manifest_path.with_name(f"{manifest_path.name}.tmp")
        with manifest_tmp.open("w", encoding="utf-8") as f:
            for line in filtered_records:
                # Rewrite semantic_code_path to local relative path
                row = json.loads(line)
                bin_name = row.get("semantic_code_path", "")
                row["semantic_code_path"] = str(dataset_bins_dir / bin_name)
                # Codes are pulled to be used: stamp what s2mel's
                # `_has_semantic_codes()` gate requires so the manifest is
                # trainable as-is, with no separate stamping pass.  Do NOT add
                # `semantic_max_audio_seconds` -- the collator rejects a batch
                # whose value disagrees with the training config, and the
                # duration filter above already covers it.
                row["semantic_lookup_path"] = str(semantic_lookup_path)
                row["semantic_lookup_sha256"] = semantic_lookup_sha256
                row["semantic_codec"] = SEMANTIC_CODEC_NAME
                row["semantic_codebooks"] = SEMANTIC_CODEBOOKS
                row["semantic_fps"] = float(row.get("semantic_frame_rate", 50.0))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest_tmp.replace(manifest_path)
        log(
            "maskgct_manifest_written",
            dataset=dataset,
            records=len(filtered_records),
            path=str(manifest_path),
            semantic_lookup_path=str(semantic_lookup_path),
            semantic_lookup_sha256=semantic_lookup_sha256,
        )

    log("MASKGCT_CODES_COMPLETE", output=str(codes_root))


def main() -> None:
    args = parse_args()
    if args.min_duration < 0:
        raise ValueError("--min-duration must be non-negative")
    if (
        args.min_sample_rate < 0
        or args.sample_metadata < 0
        or args.metadata_shard_sample < 0
    ):
        raise ValueError("sample-rate and sampling arguments must be non-negative")
    if args.workers < 1 or args.metadata_workers < 1 or args.attempts < 1:
        raise ValueError("--workers, --metadata-workers, and --attempts must be positive")
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("--datasets contains duplicate names")
    if args.summary_file.name != str(args.summary_file):
        raise ValueError("--summary-file must be a filename under output-root")
    if args.metadata_shard_sample and not args.preflight_only:
        raise ValueError("--metadata-shard-sample requires --preflight-only")

    key_file = args.key_file.expanduser().resolve()
    if not key_file.is_file():
        raise FileNotFoundError(f"GCS key file not found: {key_file}")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_file)
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT
    fs = gcsfs.GCSFileSystem(project=PROJECT, token=str(key_file))

    filter_key = _filter_params_key(args)
    sync_state = load_sync_state(output_root) if not args.force_rescan else {}

    # Fail before any scanning or downloading if the codes cannot be made usable.
    semantic_lookup_path: Path | None = None
    semantic_lookup_sha256 = ""
    if not args.no_pull_maskgct_codes:
        semantic_lookup_path, semantic_lookup_sha256 = resolve_semantic_lookup(
            output_root, args.semantic_lookup
        )
        log(
            "semantic_lookup",
            path=str(semantic_lookup_path),
            sha256=semantic_lookup_sha256,
            max_audio_seconds=args.max_audio_seconds,
        )

    log(
        "start",
        source=f"gs://{SOURCE_PREFIX}",
        output_root=str(output_root),
        datasets=args.datasets,
        min_duration=args.min_duration,
        min_sample_rate=args.min_sample_rate,
        metadata_workers=args.metadata_workers,
        workers=args.workers,
        incremental=not args.force_rescan,
        filter_key=filter_key,
    )

    # Incremental detection: check which datasets have new tar shards
    datasets_to_scan: list[str] = []
    prefetched_audio_details: dict[str, dict[str, dict[str, Any]]] = {}
    skipped_datasets: list[str] = []
    partitioned_datasets = {
        ds.strip() for ds in args.partitioned_datasets.split(",") if ds.strip()
    }
    code_shard_stems: dict[str, set[str]] = {}
    if args.resume_dataset:
        if args.datasets != [args.resume_dataset]:
            raise ValueError(
                "--resume-dataset must be the only entry in --datasets"
            )
        if args.preflight_only:
            raise ValueError(
                "--preflight-only is not supported with --resume-dataset"
            )
        resumed_plan, resumed_audio_details = build_plan_from_metadata_tmp(
            output_root,
            args.resume_dataset,
            args.resume_dataset in partitioned_datasets,
            fs,
            args.attempts,
        )
        plans = [resumed_plan]
        prefetched_audio_details = {
            args.resume_dataset: resumed_audio_details
        }
        log(
            "resume_mode",
            dataset=args.resume_dataset,
            partitioned=args.resume_dataset in partitioned_datasets,
        )
    else:
        code_shards_only = (
            args.codes_shards_only and not args.no_require_maskgct_codes
        )
        if code_shards_only:
            for dataset in args.datasets:
                code_shard_stems[dataset] = list_code_shard_stems(
                    fs, dataset, args.attempts
                )
                log(
                    "code_shards_listed",
                    dataset=dataset,
                    code_shards=len(code_shard_stems[dataset]),
                )
        for dataset in args.datasets:
            audio_details, new_tars, needs_rescan = detect_new_tars(
                fs, dataset, args.attempts, sync_state, filter_key,
                code_shard_stems.get(dataset) if code_shards_only else None,
            )
            if needs_rescan:
                datasets_to_scan.append(dataset)
                prefetched_audio_details[dataset] = audio_details
            else:
                skipped_datasets.append(dataset)

        if skipped_datasets:
            log(
                "incremental_skipped",
                datasets=skipped_datasets,
                reason="no new tar shards and filters unchanged",
            )

        if not datasets_to_scan:
            log("ALL_UP_TO_DATE", message="no new data detected, nothing to do")
            return

        plans = [
            scan_dataset(
                fs,
                dataset,
                args.min_duration,
                args.min_sample_rate,
                args.metadata_shard_sample,
                args.sample_seed,
                args.metadata_workers,
                args.attempts,
                max_cer=args.max_cer,
                asr_primary=args.asr_primary,
                asr_secondary=args.asr_secondary,
                skip_cer=args.no_cer_filter,
                require_maskgct_codes=not args.no_require_maskgct_codes,
                languages=(
                    set(args.languages.lower().split(","))
                    if args.languages
                    else None
                ),
                max_hours_per_lang=args.max_hours_per_lang,
                audio_details=prefetched_audio_details.get(dataset),
                partitioned=dataset in partitioned_datasets,
                code_shard_stems=code_shard_stems.get(dataset),
            )
            for dataset in datasets_to_scan
        ]

        # Speaker count filtering
        if not args.no_speaker_filter and args.min_speaker_records > 0:
            for plan in plans:
                speaker_counts: dict[str, int] = defaultdict(int)
                for selected in plan.selected:
                    speaker_id = selected.metadata.get("speaker_id")
                    if isinstance(speaker_id, str) and speaker_id:
                        speaker_counts[speaker_id] += 1
                removed_speakers = {
                    sid for sid, count in speaker_counts.items()
                    if count < args.min_speaker_records
                }
                if removed_speakers:
                    before_count = len(plan.selected)
                    kept = []
                    removed_tars: set[tuple[str, str, int]] = set()
                    for selected in plan.selected:
                        speaker_id = selected.metadata.get("speaker_id")
                        if (
                            isinstance(speaker_id, str)
                            and speaker_id in removed_speakers
                        ):
                            removed_tars.add((
                                next(
                                    tar_key
                                    for tar_key, members in plan.by_tar.items()
                                    if selected.member in members
                                    and selected.member_occurrence
                                    in members[selected.member]
                                ),
                                selected.member,
                                selected.member_occurrence,
                            ))
                            duration = selected.metadata.get("duration", 0)
                            plan.selected_duration_seconds -= float(duration)
                        else:
                            kept.append(selected)
                    plan.selected = kept
                    # Clean up by_tar
                    for tar_key, member, occurrence in removed_tars:
                        if member in plan.by_tar.get(tar_key, {}):
                            plan.by_tar[tar_key][member].pop(occurrence, None)
                            if not plan.by_tar[tar_key][member]:
                                del plan.by_tar[tar_key][member]
                        if tar_key in plan.by_tar and not plan.by_tar[tar_key]:
                            del plan.by_tar[tar_key]
                    log(
                        "speaker_filter",
                        dataset=plan.name,
                        removed_speakers=len(removed_speakers),
                        records_before=before_count,
                        records_after=len(plan.selected),
                        min_speaker_records=args.min_speaker_records,
                    )
        if args.sample_metadata:
            rng = random.Random(args.sample_seed)
            reservoir: list[tuple[str, dict[str, Any]]] = []
            seen = 0
            for plan in plans:
                for selected in plan.selected:
                    seen += 1
                    candidate = (plan.name, selected.metadata)
                    if len(reservoir) < args.sample_metadata:
                        reservoir.append(candidate)
                        continue
                    replacement = rng.randrange(seen)
                    if replacement < args.sample_metadata:
                        reservoir[replacement] = candidate
            for sample_index, (dataset, metadata) in enumerate(
                reservoir, start=1
            ):
                log(
                    "metadata_sample",
                    sample_index=sample_index,
                    sample_seed=args.sample_seed,
                    dataset=dataset,
                    metadata=metadata,
                )
    referenced_tar_bytes = sum(plan.referenced_tar_bytes for plan in plans)
    disk = shutil.disk_usage(output_root)
    if referenced_tar_bytes + RESERVED_FREE_BYTES > disk.free:
        raise OSError(
            "Insufficient free space under conservative referenced-tar bound: "
            f"free={disk.free}, referenced_tars={referenced_tar_bytes}, "
            f"reserved={RESERVED_FREE_BYTES}"
        )
    log(
        "PREFLIGHT_COMPLETE",
        datasets=len(plans),
        metadata_shards=sum(plan.metadata_shards for plan in plans),
        source_records=sum(plan.source_records for plan in plans),
        selected_records=sum(len(plan.selected) for plan in plans),
        selected_duration_hours=round(
            sum(plan.selected_duration_seconds for plan in plans) / 3600.0,
            3,
        ),
        min_sample_rate=args.min_sample_rate,
        referenced_tars=sum(len(plan.by_tar) for plan in plans),
        referenced_tar_bytes=referenced_tar_bytes,
        free_bytes=disk.free,
        reserved_free_bytes=RESERVED_FREE_BYTES,
    )
    if args.preflight_only:
        if skipped_datasets:
            log("PREFLIGHT_NOTE", message=f"{len(skipped_datasets)} datasets skipped (no new tars)")
        if not args.no_pull_maskgct_codes:
            log("PREFLIGHT_NOTE", message="maskGCT codes will be downloaded after sync")
        return

    metadata_tmp_paths = {
        plan.name: write_metadata_tmp(output_root, plan) for plan in plans
    }
    log("METADATA_TMP_COMPLETE", datasets=len(metadata_tmp_paths))

    summaries: list[dict[str, Any]] = []
    for plan in plans:
        if args.resume_dataset:
            extract_dataset_resume(
                fs,
                output_root,
                plan,
                args.attempts,
                args.workers,
            )
        else:
            extract_dataset(
                fs,
                output_root,
                plan,
                args.attempts,
                args.workers,
            )
        summaries.append(
            verify_and_finalize_dataset(
                output_root,
                plan,
                metadata_tmp_paths[plan.name],
                args.min_duration,
                args.min_sample_rate,
            )
        )

    # Download maskGCT codes if requested
    if not args.no_pull_maskgct_codes:
        assert semantic_lookup_path is not None  # resolved above
        download_maskgct_codes(
            fs,
            output_root,
            plans,
            args.attempts,
            args.workers,
            semantic_lookup_path=semantic_lookup_path,
            semantic_lookup_sha256=semantic_lookup_sha256,
            max_audio_seconds=args.max_audio_seconds,
        )

    # Update sync state with successfully synced tar shards
    updated_state = load_sync_state(output_root)
    if "datasets" not in updated_state:
        updated_state["datasets"] = {}
    for plan in plans:
        current_tars = sorted(prefetched_audio_details.get(plan.name, {}).keys())
        dataset_state = {
            "filter_key": filter_key,
            "synced_tars": current_tars,
            "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selected_records": len(plan.selected),
        }
        # Codes keep being encoded upstream, so the coded-shard set is what the
        # next incremental run diffs against.
        if plan.name in code_shard_stems:
            dataset_state["synced_code_shards"] = sorted(code_shard_stems[plan.name])
        updated_state["datasets"][plan.name] = dataset_state
    save_sync_state(output_root, updated_state)
    log("SYNC_STATE_SAVED", datasets=len(plans))

    summary = {
        "source": f"gs://{SOURCE_PREFIX}",
        "output_root": str(output_root),
        "duration_filter": f"> {args.min_duration}",
        "sample_rate_filter": f">= {args.min_sample_rate}",
        "cer_filter": f"<= {args.max_cer}" if not args.no_cer_filter else "disabled",
        "speaker_filter": f">= {args.min_speaker_records} records" if not args.no_speaker_filter else "disabled",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": summaries,
        "totals": {
            "datasets": len(summaries),
            "selected_records": sum(
                item["selected_records"] for item in summaries
            ),
            "selected_duration_seconds": sum(
                item["selected_duration_seconds"] for item in summaries
            ),
            "selected_duration_hours": sum(
                item["selected_duration_hours"] for item in summaries
            ),
            "output_bytes": sum(item["output_bytes"] for item in summaries),
        },
    }
    atomic_write_json(output_root / args.summary_file, summary)
    log("ALL_COMPLETE", **summary["totals"])


if __name__ == "__main__":
    main()
