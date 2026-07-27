# maskgct_codes_io

从 `SpeechData` 同步过来的两个脚本，用来消费 bucket 里的 MaskGCT semantic codes。
格式说明见 `docs/maskgct-codes-bucket-format.md`。**上游是
`/mnt/data_sdd/hhy/SpeechData/features/`，改动请改上游再同步，不要只改这里。**

- `read_codes.py` — `CodesShard` 类，按样本 id 取 codes（memmap `<u2`）。
  独立可用：`python read_codes.py --dataset vctk --shard vctk-000000`
- `build_manifest.py` — 把 bucket 索引转成本项目训练侧要的 manifest
  （`semantic_fps`、可解析的 `semantic_code_path`、`semantic_lookup_path`/`_sha256`），
  并做与训练侧对齐的过滤。

`--emit-lookup` 要生成 `maskgct_lookup.pt`，需要 RepCodec 权重和
`SpeechData/features/semantic_codec.py`，所以那一步要在 SpeechData 里跑：

```bash
cd /mnt/data_sdd/hhy/SpeechData
python features/build_manifest.py --dataset vctk --out /data/manifests/vctk.jsonl \
    --emit-lookup
```

已有 lookup 表时，这里就能直接跑（`--lookup-path` 复用，不需要 codec 权重）：

```bash
python scripts/maskgct_codes_io/build_manifest.py --dataset vctk \
    --out /data/manifests/vctk.jsonl --emit-lookup \
    --lookup-path /data/manifests/maskgct_lookup.pt
```
