#!/usr/bin/env python3
"""Build the local duration-filtered s2mel training mirror from GCS shards."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
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
T = TypeVar("T")


def log(event: str, **fields: Any) -> None:
    print(
        json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between two strings."""
    if not reference and not hypothesis:
        return 0.0
    if not reference:
        return 1.0
    ref = list(reference.strip())
    hyp = list(hypothesis.strip())
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / n if n > 0 else 0.0


def load_asr_texts(
    fs: gcsfs.GCSFileSystem,
    dataset_prefix: str,
    asr_model: str,
    attempts: int,
) -> dict[str, str]:
    """Load ASR transcriptions for a dataset into a dict keyed by record id."""
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
    if not asr_paths:
        return {}
    texts: dict[str, str] = {}
    for asr_path in asr_paths:
        try:
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
        except Exception as exc:
            log("asr_load_warning", asr_path=asr_path, error=str(exc))
    return texts


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


@dataclass
class SelectedRecord:
    metadata: dict[str, Any]
    member: str
    member_occurrence: int
    basename: str
    expected_size: int | None = None


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
    audio_details: dict[str, dict[str, Any]] | None = None,
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

    # Load ASR texts for CER filtering
    asr_primary_texts: dict[str, str] = {}
    asr_secondary_texts: dict[str, str] = {}
    if not skip_cer:
        dataset_prefix = f"{SOURCE_PREFIX}/{dataset}"
        asr_primary_texts = load_asr_texts(fs, dataset_prefix, asr_primary, attempts)
        asr_secondary_texts = load_asr_texts(fs, dataset_prefix, asr_secondary, attempts)
        log(
            "asr_loaded",
            dataset=dataset,
            primary_model=asr_primary,
            primary_records=len(asr_primary_texts),
            secondary_model=asr_secondary,
            secondary_records=len(asr_secondary_texts),
        )
    source_metadata_shards = len(metadata_paths)
    if metadata_shard_sample and metadata_shard_sample < len(metadata_paths):
        metadata_paths = sorted(
            random.Random(f"{sample_seed}:{dataset}").sample(
                metadata_paths,
                metadata_shard_sample,
            )
        )

    plan = DatasetPlan(name=dataset, metadata_shards=len(metadata_paths))
    seen_ids: dict[str, str] = {}
    seen_basenames: dict[str, str] = {}
    member_occurrences: dict[tuple[str, str], int] = defaultdict(int)

    for metadata_path, rows in iter_metadata_rows_bounded(
        fs,
        metadata_paths,
        attempts,
        metadata_workers,
    ):
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
                raise FileNotFoundError(
                    f"{location}: referenced tar does not exist: gs://{tar_key}"
                )
            occurrence_key = (tar_key, member)
            member_occurrence = member_occurrences[occurrence_key]
            member_occurrences[occurrence_key] += 1

            if duration <= min_duration or sample_rate < min_sample_rate:
                continue

            # CER filtering
            if not skip_cer:
                record_id_for_cer = row.get("id", "")
                metadata_text = row.get("text")
                has_text = isinstance(metadata_text, str) and metadata_text.strip()
                if has_text:
                    # Use metadata text vs primary ASR
                    asr_text = asr_primary_texts.get(record_id_for_cer)
                    if asr_text:
                        cer = compute_cer(metadata_text.strip(), asr_text)
                        if cer > max_cer:
                            continue
                else:
                    # Use primary ASR vs secondary ASR
                    primary_text = asr_primary_texts.get(record_id_for_cer)
                    secondary_text = asr_secondary_texts.get(record_id_for_cer)
                    if primary_text and secondary_text:
                        cer = compute_cer(primary_text, secondary_text)
                        if cer > max_cer:
                            continue
                    elif not primary_text and not secondary_text:
                        # No ASR available and no text — skip
                        continue

            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{location}: missing id")
            if record_id in seen_ids:
                raise ValueError(
                    f"{dataset}: duplicate selected id {record_id!r}: "
                    f"{seen_ids[record_id]} and {location}"
                )

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
            output_metadata["audio_path"] = f"../audios/{dataset}/{basename}"
            selected = SelectedRecord(
                metadata=output_metadata,
                member=member,
                member_occurrence=member_occurrence,
                basename=basename,
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
) -> None:
    source = tar_file.extractfile(member_info)
    if source is None:
        raise FileNotFoundError(f"Could not extract tar member {member_info.name}")
    tmp_path = destination.with_name(f".{destination.name}.part")
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
                destination = dataset_dir / selected.basename
                if valid_flac(destination, member_info.size):
                    reused += 1
                else:
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
) -> int | None:
    sizes: list[tuple[SelectedRecord, int]] = []
    for records_by_occurrence in selected_by_member.values():
        for selected in records_by_occurrence.values():
            path = dataset_dir / selected.basename
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
        reused = reuse_complete_shard(selected_by_member, dataset_dir)
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
    ) -> tuple[int, int]:
        return retry(
            f"extract gs://{tar_key}",
            lambda: extract_tar_once(
                fs,
                tar_key,
                selected_by_member,
                dataset_dir,
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


def verify_and_finalize_dataset(
    output_root: Path,
    plan: DatasetPlan,
    metadata_tmp: Path,
    min_duration: float,
    min_sample_rate: int,
) -> dict[str, Any]:
    dataset_dir = output_root / "audios" / plan.name
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
        path = dataset_dir / selected.basename
        if selected.expected_size is None or not valid_flac(
            path, selected.expected_size
        ):
            raise ValueError(f"{plan.name}: invalid output FLAC: {path}")
        total_bytes += path.stat().st_size

    expected_audio_paths = {
        f"../audios/{plan.name}/{selected.basename}" for selected in plan.selected
    }
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
            expected = (dataset_dir / PurePosixPath(audio_path).name).resolve()
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
    parser.add_argument("--no-pull-maskgct-codes", action="store_true",
                        help="Skip downloading maskGCT semantic codes from GCS.")
    parser.add_argument("--force-rescan", action="store_true",
                        help="Ignore sync state and re-scan all datasets from scratch.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()



def download_maskgct_codes(
    fs: gcsfs.GCSFileSystem,
    output_root: Path,
    plans: list,
    attempts: int,
    workers: int,
) -> None:
    """Download maskGCT semantic codes and write filtered manifests."""
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
        )

        # Download needed bin shards
        dataset_bins_dir = bins_dir / dataset
        dataset_bins_dir.mkdir(parents=True, exist_ok=True)

        for bin_file in bin_files:
            bin_name = bin_file.rsplit("/", 1)[-1]
            if bin_name not in needed_bins:
                continue
            target = dataset_bins_dir / bin_name
            if target.is_file() and target.stat().st_size > 0:
                continue
            tmp = target.with_name(f".{target.name}.tmp")
            try:
                with retry(
                    f"download {bin_file}",
                    lambda p=bin_file: fs.open(p, "rb"),
                    attempts,
                ) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp.replace(target)
                log("maskgct_bin_downloaded", dataset=dataset, file=bin_name,
                    size_mb=round(target.stat().st_size / 1024 / 1024, 1))
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                log("maskgct_bin_error", dataset=dataset, file=bin_name, error=str(exc))

        # Write filtered manifest
        manifest_path = manifests_dir / f"{dataset}.jsonl"
        manifest_tmp = manifest_path.with_name(f"{manifest_path.name}.tmp")
        with manifest_tmp.open("w", encoding="utf-8") as f:
            for line in filtered_records:
                # Rewrite semantic_code_path to local relative path
                row = json.loads(line)
                bin_name = row.get("semantic_code_path", "")
                row["semantic_code_path"] = str(dataset_bins_dir / bin_name)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest_tmp.replace(manifest_path)
        log(
            "maskgct_manifest_written",
            dataset=dataset,
            records=len(filtered_records),
            path=str(manifest_path),
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
    for dataset in args.datasets:
        audio_details, new_tars, needs_rescan = detect_new_tars(
            fs, dataset, args.attempts, sync_state, filter_key,
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
            audio_details=prefetched_audio_details.get(dataset),
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
                    if isinstance(speaker_id, str) and speaker_id in removed_speakers:
                        removed_tars.add((
                            next(
                                tar_key
                                for tar_key, members in plan.by_tar.items()
                                if selected.member in members
                                and selected.member_occurrence in members[selected.member]
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
        for sample_index, (dataset, metadata) in enumerate(reservoir, start=1):
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
        download_maskgct_codes(fs, output_root, plans, args.attempts, args.workers)

    # Update sync state with successfully synced tar shards
    updated_state = load_sync_state(output_root)
    if "datasets" not in updated_state:
        updated_state["datasets"] = {}
    for plan in plans:
        current_tars = sorted(prefetched_audio_details.get(plan.name, {}).keys())
        updated_state["datasets"][plan.name] = {
            "filter_key": filter_key,
            "synced_tars": current_tars,
            "synced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selected_records": len(plan.selected),
        }
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
