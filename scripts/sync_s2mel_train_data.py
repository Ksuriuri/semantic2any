#!/usr/bin/env python3
"""Build the local duration-filtered s2mel training mirror from GCS shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
DEFAULT_MIN_DURATION = 6.0
DEFAULT_WORKERS = 4
RESERVED_FREE_BYTES = 20 * 1024**3
COPY_CHUNK_BYTES = 8 * 1024**2
T = TypeVar("T")


def log(event: str, **fields: Any) -> None:
    print(
        json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


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


def scan_dataset(
    fs: gcsfs.GCSFileSystem,
    dataset: str,
    min_duration: float,
    attempts: int,
) -> DatasetPlan:
    dataset_prefix = f"{SOURCE_PREFIX}/{dataset}"
    metadata_paths = glob_paths(
        fs,
        f"{dataset_prefix}/metadata/*.jsonl",
        attempts,
    )
    audio_details = glob_details(
        fs,
        f"{dataset_prefix}/audio/*.tar",
        attempts,
    )
    if not metadata_paths:
        raise FileNotFoundError(f"No metadata shards found for {dataset}")
    if not audio_details:
        raise FileNotFoundError(f"No audio tar shards found for {dataset}")

    plan = DatasetPlan(name=dataset, metadata_shards=len(metadata_paths))
    seen_ids: dict[str, str] = {}
    seen_basenames: dict[str, str] = {}
    member_occurrences: dict[tuple[str, str], int] = defaultdict(int)

    for metadata_path in metadata_paths:
        rows = retry(
            f"read metadata {metadata_path}",
            lambda metadata_path=metadata_path: read_metadata_rows(fs, metadata_path),
            attempts,
        )
        for line_number, row in rows:
            plan.source_records += 1
            location = f"{metadata_path}:{line_number}"
            raw_duration = row.get("duration")
            if isinstance(raw_duration, bool) or not isinstance(
                raw_duration, (int, float)
            ):
                raise ValueError(f"{location}: duration is missing or non-numeric")
            duration = float(raw_duration)

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

            if duration <= min_duration:
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
            output_metadata["audio_path"] = f"../{dataset}/{basename}"
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
        source_records=plan.source_records,
        selected_records=len(plan.selected),
        selected_duration_hours=round(plan.selected_duration_seconds / 3600.0, 3),
        referenced_tars=len(plan.by_tar),
        referenced_tar_bytes=plan.referenced_tar_bytes,
    )
    return plan


def write_metadata_tmp(output_root: Path, plan: DatasetPlan) -> Path:
    (output_root / plan.name).mkdir(parents=True, exist_ok=True)
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
    dataset_dir = output_root / plan.name
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
) -> dict[str, Any]:
    dataset_dir = output_root / plan.name
    expected_names = {selected.basename for selected in plan.selected}
    actual_names = {
        path.name for path in dataset_dir.glob("*.flac") if path.is_file()
    }
    if expected_names != actual_names:
        missing = sorted(expected_names - actual_names)
        extras = sorted(actual_names - expected_names)
        raise ValueError(
            f"{plan.name}: output FLAC set mismatch: "
            f"missing={missing[:5]} ({len(missing)}), "
            f"extras={extras[:5]} ({len(extras)})"
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
        f"../{plan.name}/{selected.basename}" for selected in plan.selected
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
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset child prefixes under gs://.../preprocessed/.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_duration < 0:
        raise ValueError("--min-duration must be non-negative")
    if args.workers < 1 or args.attempts < 1:
        raise ValueError("--workers and --attempts must be positive")
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("--datasets contains duplicate names")

    key_file = args.key_file.expanduser().resolve()
    if not key_file.is_file():
        raise FileNotFoundError(f"GCS key file not found: {key_file}")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(key_file)
    os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT
    fs = gcsfs.GCSFileSystem(project=PROJECT, token=str(key_file))
    log(
        "start",
        source=f"gs://{SOURCE_PREFIX}",
        output_root=str(output_root),
        datasets=args.datasets,
        min_duration=args.min_duration,
        workers=args.workers,
    )

    plans = [
        scan_dataset(
            fs,
            dataset,
            args.min_duration,
            args.attempts,
        )
        for dataset in args.datasets
    ]
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
        referenced_tars=sum(len(plan.by_tar) for plan in plans),
        referenced_tar_bytes=referenced_tar_bytes,
        free_bytes=disk.free,
        reserved_free_bytes=RESERVED_FREE_BYTES,
    )
    if args.preflight_only:
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
            )
        )

    summary = {
        "source": f"gs://{SOURCE_PREFIX}",
        "output_root": str(output_root),
        "duration_filter": f"> {args.min_duration}",
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
    atomic_write_json(output_root / "download_summary.json", summary)
    log("ALL_COMPLETE", **summary["totals"])


if __name__ == "__main__":
    main()
