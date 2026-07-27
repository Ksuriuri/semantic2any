# bucket 里的 MaskGCT semantic codes：布局、读法、过滤

产出方是 `SpeechData/features/maskgct_codes.py`（本机 `/mnt/data_sdd/hhy/SpeechData`）。
本文说明本项目该怎么消费它。**结论：两个项目的读取代码本来就是对的，缺的只是
manifest 字段的拼写和路径，用 `SpeechData/features/build_manifest.py` 生成 manifest 即可，
不需要改训练侧代码。**

## 1. 布局

每个音频 tar 出一对文件，在
`gs://noiz-taiwan-audio-data/preprocessed/<dataset>/features/maskGCT_codes/`：

```text
<dataset>-NNNNNN.u2.bin    该 tar 全部样本的 codes 拼接，little-endian uint16
<dataset>-NNNNNN.jsonl     每条样本一行，带在 .bin 里的 offset / length
errors/  _leases/
```

一条样本的 codes 就是 `blob[offset : offset+length]`，**offset/length 单位是 code 数、
不是字节**，所以 `np.memmap(path, dtype="<u2")` 可以直接用索引里的数字切片
——这正是两个项目现有的做法：

- `text2semantic/finetuning/dataset.py` → `_semantic_codes()`
- `semantic2any/semantic2any/data/s2mel_dataset.py` → `_load_semantic_codes()`

50 Hz × uint16 = **每秒音频 100 字节**。7 个数据集 1927 音频小时，codes 总共约 0.7 GB，
但样本 120 万条——所以打包而不是一条一个对象。

## 2. 一定要先整块拉到本地再 mmap

实测 vctk-000000（2000 条）：

| 方式 | 单条耗时 |
| --- | --- |
| 本地 mmap 随机读 | **1.5 µs** |
| GCS 逐条 ranged GET | **513 ms** |

差约 34 万倍（整块下载 1.94 s）。一个 shard 一个请求，不要一条音频一个请求。

## 3. 索引字段 → 训练侧字段（曾经的三个不一致）

bucket 索引是**共享数据底座**的口径：全量、不过滤、路径可搬迁。训练侧要的稍有不同，
差异有三处，全部在写入侧解决：

| bucket 索引 | 训练侧期望 | 差异与处理 |
| --- | --- | --- |
| `semantic_frame_rate` = 50.0 | `s2mel_dataset` 读 `semantic_fps` | manifest 里两个都写。注意 `_record_has_singleton_semantic_budget` 是 `record.get("semantic_fps", 50.0)`，默认值恰好也是 50，所以旧口径**不会报错、只是靠巧合正确** |
| `semantic_code_path` = 同目录**文件名** | `t2s` 把字符串**原样**传 `np.memmap`（不解析相对路径）；`s2any` 用 `_resolve_path()` 相对 **manifest 所在目录**解析 | `--path-style absolute` 给 t2s，`relative`（默认）给 s2any |
| 无 lookup 字段 | `s2any._has_semantic_codes()` 要求 `semantic_lookup_path` + `semantic_lookup_sha256`，且一个 batch 只能有一个（`s2mel_dataset.py:1219` 会 raise） | `--emit-lookup` 从 codec checkpoint 生成 `maskgct_lookup.pt`（8192×1024 float32）并把 sha256 打到每一行 |

lookup 表是 `quantizer.vq2emb(arange(8192))`，只取决于 RepCodec 权重、与数据无关，
所以它不该进每个 tar 的 shard，而是每棵 manifest 树生成一次。与
`semantic2any/scripts/precompute_maskgct_codes.py` 写的那张表同构。

其余 `id` / `duration` / `speaker_id` / `language` / `sample_rate` / `audio_path` 从 metadata
带过来了，拼 manifest 不必再读 metadata。`status` 为 `skipped_short_audio` 的行没有
offset（见 §5），`build_manifest.py` 会跳过。

## 4. 生成 manifest

```bash
cd /mnt/data_sdd/hhy/SpeechData

# semantic2any（相对路径 + lookup 表）
python features/build_manifest.py --dataset vctk --out /data/manifests/vctk.jsonl \
    --emit-lookup

# text2semantic（绝对路径；它不解析相对路径），带它自己那套过滤
python features/build_manifest.py --dataset vctk --out /data/manifests/vctk.jsonl \
    --path-style absolute --max-target-seconds 30 --require-speaker-id \
    --min-speaker-records 2
```

只读单个 shard 看一眼：`python features/read_codes.py --dataset vctk --shard vctk-000000`。

## 5. 过滤规则：bucket 不过滤，manifest 才过滤

**bucket 对 tar 里每一条样本都出 codes**，不只训练过滤后的子集——bucket 是共享底座，
过滤是某一次训练的选择；而且 tar 是整个下载的，只编码子集不省流量，只省约 20% GPU 时间。

生成侧唯一的例外是 `--min-audio-duration-seconds 0.05`：WutheringWaves 有 **32 条单帧
44.1 kHz 音频**（时长 2.3e-5 s，98 字节 flac），任何模型都编不出来，而"有失败样本就不写
索引"会让这些 tar 永远重跑。这 32 条记 `skipped_short_audio`。阈值安全是因为它们全在
0.01 s 以下，而 7 个数据集里次短的 clip 是 0.098 s。

训练侧过滤（`build_manifest.py` 的参数与两个项目自己的实现对齐，只会去掉它们本来也会
丢弃的行）：

| 参数 | 对应训练侧 |
| --- | --- |
| `--max-target-seconds` | `Text2SemanticDataset.max_target_seconds`（其默认 30.0） |
| `--max-semantic-tokens` | `max_semantic_tokens`，比的是 `semantic_code_length` |
| `--min-target-seconds` | s2any 的 `min_target_seconds` 一类时长下限 |
| `--require-speaker-id` | t2s 需要说话人参考音频，`speaker_id` 缺失就用不了 |
| `--min-speaker-records` | `min_speaker_records`（默认 2：要能配出参考 clip） |

注意上游导出把缺失值写成**字符串 `"None"`**，不是 `null`；`build_manifest.py` 的
`--require-speaker-id` 已按这个处理。s2any 侧另有配对相关的门槛
（`min_prompt_seconds` / `max_prompt_seconds` / `min_generated_frames` / `max_pair_seconds`），
那些依赖 prompt/target 的配对结果，仍留在 dataset 里做。

## 6. 长音频：不要按波形切窗

w2v-bert 的 `relative_key` attention 会一次性构造 L×L×64 的位置张量，162 s 的 clip 单次
申请 15.68 GiB → OOM。修法是 `features/semantic_codec.py` 的
`patch_relative_key_attention()`，**按 query 行分块算 attention**，数学等价、实测逐位相同。
**不要改成切波形**：实测重叠切窗只有 34–46% 的 code 与单次 forward 一致，
`relative_key` 是全局 attention，切输入必然改结果。

这条对训练侧的意义：bucket 里的 codes 是**整段音频一次 forward** 的结果，
所以推理/微调时若自己重新编码长音频，必须用同一份 `semantic_codec.py`，否则 code 不一致。
