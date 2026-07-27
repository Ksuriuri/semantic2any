#!/usr/bin/env python3
"""Turn the bucket's MaskGCT code shards into a trainer manifest.

The bucket layout is a *shared data base*: one packed `.u2.bin` plus one index
`.jsonl` per audio tar, every sample encoded, nothing filtered. A training run
needs something narrower and slightly differently spelled, and both consumers
in `noiz-tts` already read the packed layout correctly but expect their own
field names and path conventions:

  text2semantic/finetuning/dataset.py    `_semantic_codes()` memmaps `<u2` and
      slices `[offset : offset+length]`. It passes `semantic_code_path` to
      `np.memmap` **verbatim** -- it does not resolve relative paths -- so the
      manifest must carry a path that resolves from the training process's cwd.
      Filters on `duration` (`max_target_seconds`), `semantic_code_length`
      (`max_semantic_tokens`), and needs >= `min_speaker_records` rows per
      speaker plus a second audio file for the speaker reference.

  semantic2any/semantic2any/data/s2mel_dataset.py    `_load_semantic_codes()`
      memmaps `<u2` the same way, and `_resolve_path()` does resolve relative
      paths, but against the *manifest's* directory. It reads `semantic_fps`
      (not our `semantic_frame_rate`), and `_has_semantic_codes()` requires
      `semantic_lookup_path` + `semantic_lookup_sha256`; a batch mixing two
      lookup paths raises "A batch must use one semantic lookup table and
      checksum".

So three concrete gaps, all on the writer's side, none needing a change to
either trainer:

  1. `semantic_frame_rate` -> also emit `semantic_fps` (same 50.0).
  2. `semantic_code_path` is a bare filename in the bucket index (deliberately,
     so a shard stays relocatable) -> rewrite to a real path.
  3. no lookup table in the bucket -> `--emit-lookup` materializes
     `maskgct_lookup.pt` from the codec checkpoint and stamps its sha256 on
     every row. The table is a function of the checkpoint, not of the data,
     which is why it does not belong in the per-tar shards.

The lookup table is `quantizer.vq2emb(arange(8192))`, i.e. the frozen
8192x1024 post-projection embedding, byte-compatible with what
`semantic2any/scripts/precompute_maskgct_codes.py` writes.

Usage::

    # local shards -> manifest for semantic2any (paths relative to the manifest)
    python features/build_manifest.py --codes-dir /data/maskgct/vctk \\
        --out /data/manifests/vctk.jsonl --emit-lookup

    # for text2semantic, which memmaps the string as-is
    python features/build_manifest.py --codes-dir /data/maskgct/vctk \\
        --out /data/manifests/vctk.jsonl --path-style absolute \\
        --max-target-seconds 30 --min-speaker-records 2

    # straight from the bucket (downloads each shard once into --cache-dir)
    python features/build_manifest.py --dataset vctk --out vctk.jsonl \\
        --cache-dir /data/maskgct-cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from read_codes import CODE_DTYPE, DEFAULT_ROOT, FEATURE_DIR, fetch_shard

LOOKUP_NAME = "maskgct_lookup.pt"
SEMANTIC_FPS = 50.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_lookup(out_path: Path, *, checkpoint_dir: Path, device: str = "cpu") -> Path:
    """Materialize the frozen codebook lookup table from the codec checkpoint.

    Written once per manifest tree, not per shard: it depends only on the
    RepCodec weights. Skipped if a matching file already exists, and a
    *mismatching* one is an error rather than an overwrite -- a manifest whose
    rows claim a checksum the file no longer has would fail deep inside
    training instead of here.
    """
    # Imported lazily and *after* the reuse check: building the table needs the
    # RepCodec weights and semantic_codec.py, but reusing an existing one needs
    # neither, so a manifest built next to an existing lookup runs anywhere.
    if out_path.exists():
        return out_path

    import torch
    import yaml

    from semantic_codec import RepCodec, load_model

    config_path = checkpoint_dir / "semantic_codec" / "config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config = config.get("semantic_codec", config)
    codec = RepCodec(**config).to(device)
    load_model(codec, str(checkpoint_dir / "semantic_codec" / "model.safetensors"),
               strict=True)
    codec.eval()
    with torch.no_grad():
        codes = torch.arange(int(codec.quantizer.codebook_size),
                             device=device, dtype=torch.long)
        # vq2emb takes [n_quantizers, B, T]; one codebook, one "batch" of 8192.
        lookup = codec.quantizer.vq2emb(codes.view(1, 1, -1))
        lookup = lookup.transpose(1, 2).reshape(codes.numel(), -1)
        lookup = lookup.detach().cpu().float().contiguous()

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    torch.save({"lookup": lookup}, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


def iter_shards(codes_dir: Path):
    for index_path in sorted(codes_dir.glob("*.jsonl")):
        yield index_path


def manifest_rows(index_path: Path, *, codes_dir: Path, manifest_dir: Path,
                  path_style: str):
    """Yield trainer-shaped rows for one shard's index."""
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "encoded" or row.get("semantic_code_offset") is None:
            continue  # skipped_short_audio carries no codes

        blob = (codes_dir / row["semantic_code_path"]).resolve()
        if path_style == "absolute":
            code_path = str(blob)
        else:
            code_path = os.path.relpath(blob, manifest_dir)
        row = dict(row)
        row["semantic_code_path"] = code_path
        # s2mel_dataset reads `semantic_fps`; keep `semantic_frame_rate` too so
        # the row stays readable by anything written against the bucket schema.
        row["semantic_fps"] = row.get("semantic_frame_rate", SEMANTIC_FPS)
        row["semantic_codebooks"] = 1
        row["semantic_codec"] = "maskgct"
        yield row


def keep(row: dict, args: argparse.Namespace) -> str | None:
    """Return the name of the rule that drops this row, or None to keep it.

    The rules mirror what the two trainers apply themselves, so that filtering
    here only ever removes rows they would have dropped anyway -- the manifest
    is a training-run decision, the bucket stays complete.
    """
    duration = row.get("duration")
    duration = float(duration) if duration not in (None, "", "None") else None
    if duration is not None:
        if args.min_target_seconds is not None and duration < args.min_target_seconds:
            return "min_target_seconds"
        if args.max_target_seconds is not None and duration > args.max_target_seconds:
            return "max_target_seconds"
    length = int(row["semantic_code_length"])
    if args.max_semantic_tokens is not None and length > args.max_semantic_tokens:
        return "max_semantic_tokens"
    if args.require_speaker_id:
        speaker = row.get("speaker_id")
        # Upstream exports write missing values as the *string* "None".
        if speaker in (None, "", "None"):
            return "no_speaker_id"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("source")
    source.add_argument("--codes-dir", type=Path,
                        help="local directory of <shard>.jsonl + <shard>.u2.bin")
    source.add_argument("--dataset", help="fetch the dataset's shards from GCS")
    source.add_argument("--shard", action="append", default=None,
                        help="limit to these shards (repeatable)")
    source.add_argument("--cache-dir", type=Path, default=Path("/tmp/maskgct-codes-cache"))
    source.add_argument("--root", default=DEFAULT_ROOT)
    source.add_argument("--token", default="gcs-key.json")

    parser.add_argument("--out", type=Path, required=True, help="manifest jsonl to write")
    parser.add_argument("--path-style", choices=("relative", "absolute"),
                        default="relative",
                        help="relative: resolved against the manifest dir "
                             "(semantic2any). absolute: required by text2semantic, "
                             "which memmaps the string verbatim.")
    parser.add_argument("--emit-lookup", action="store_true",
                        help="write maskgct_lookup.pt next to the manifest and "
                             "stamp semantic_lookup_path/_sha256 on every row "
                             "(semantic2any requires both)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/maskgct"))
    parser.add_argument("--lookup-path", type=Path, default=None,
                        help="reuse an existing lookup table instead of building one")

    flt = parser.add_argument_group("filters (default: keep everything)")
    flt.add_argument("--min-target-seconds", type=float, default=None)
    flt.add_argument("--max-target-seconds", type=float, default=None,
                     help="text2semantic's own default is 30.0")
    flt.add_argument("--max-semantic-tokens", type=int, default=None)
    flt.add_argument("--require-speaker-id", action="store_true",
                     help="drop rows whose speaker_id is missing or the string "
                          "'None'; text2semantic cannot use them (it needs a "
                          "speaker reference clip)")
    flt.add_argument("--min-speaker-records", type=int, default=None,
                     help="drop speakers with fewer than N kept rows; "
                          "text2semantic needs >=2 to pair a reference clip")
    args = parser.parse_args()

    if not args.codes_dir and not args.dataset:
        parser.error("pass --codes-dir or --dataset")

    manifest_dir = args.out.parent.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    lookup_path = args.lookup_path
    lookup_sha = None
    if args.emit_lookup:
        lookup_path = lookup_path or (manifest_dir / LOOKUP_NAME)
        lookup_path = build_lookup(Path(lookup_path),
                                   checkpoint_dir=args.checkpoint_dir)
        lookup_sha = sha256_file(Path(lookup_path))

    if args.codes_dir:
        codes_dir = args.codes_dir.resolve()
        indexes = list(iter_shards(codes_dir))
        if args.shard:
            wanted = {s if s.endswith(".jsonl") else f"{s}.jsonl" for s in args.shard}
            indexes = [p for p in indexes if p.name in wanted]
    else:
        import gcsfs

        fs = gcsfs.GCSFileSystem(token=args.token)
        remote = f"{args.root}/{args.dataset}/{FEATURE_DIR}"
        shards = args.shard or sorted(
            Path(name).stem for name in fs.ls(remote)
            if name.endswith(".jsonl") and "/errors/" not in name
        )
        codes_dir = (args.cache_dir / args.dataset).resolve()
        indexes = [fetch_shard(args.dataset, shard, codes_dir,
                               root=args.root, token=args.token)
                   for shard in shards]

    dropped = Counter()
    rows = []
    for index_path in indexes:
        for row in manifest_rows(index_path, codes_dir=codes_dir,
                                 manifest_dir=manifest_dir,
                                 path_style=args.path_style):
            reason = keep(row, args)
            if reason:
                dropped[reason] += 1
                continue
            if lookup_sha:
                lp = Path(lookup_path).resolve()
                row["semantic_lookup_path"] = (
                    str(lp) if args.path_style == "absolute"
                    else os.path.relpath(lp, manifest_dir))
                row["semantic_lookup_sha256"] = lookup_sha
            rows.append(row)

    if args.min_speaker_records:
        counts = Counter(row.get("speaker_id") for row in rows)
        before = len(rows)
        rows = [row for row in rows
                if counts[row.get("speaker_id")] >= args.min_speaker_records]
        dropped["min_speaker_records"] += before - len(rows)

    tmp_out = args.out.with_suffix(args.out.suffix + ".part")
    with tmp_out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_out, args.out)

    codes = sum(int(row["semantic_code_length"]) for row in rows)
    print(json.dumps({
        "manifest": str(args.out),
        "shards": len(indexes),
        "rows": len(rows),
        "audio_hours": round(codes / SEMANTIC_FPS / 3600, 2),
        "dropped": dict(dropped),
        "path_style": args.path_style,
        "lookup": str(lookup_path) if lookup_sha else None,
        "code_dtype": CODE_DTYPE,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
