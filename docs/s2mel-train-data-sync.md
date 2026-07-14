# S2Mel 长音频数据同步

本文记录 `/mnt/data_3t_1/datasets/preprocess/s2mel-train-data` 的来源、
筛选规则、目录结构和新机器同步方法。同步入口是
`scripts/sync_s2mel_train_data.py`，固定训练/验证划分入口是
`scripts/split_s2mel_validation.py`。

## 数据来源

- GCP 项目：`noiz-430406`
- GCS 前缀：`gs://noiz-taiwan-audio-data/preprocessed/`
- 数据格式：SpeechData tar shard 与对应 JSONL metadata shard
- 数据集：
  - `ears`
  - `expresso`
  - `Genshin`
  - `hi_fi_tts`
  - `noiz-short`
  - `StarRail`
  - `vctk`
  - `WutheringWaves`

认证方式与 bucket 结构另见
[gcs-datasets.md](gcs-datasets.md)。不要把 service account key 提交到 Git。

## 处理规则

1. 读取每个数据集的全部 `metadata/*.jsonl`。
2. 严格保留数值字段 `duration > 6.0` 的记录。
3. 根据原始 `audio_path` 找到 tar shard 和 tar member。
4. 每个 tar 只读取一次，4 路并发提取所需 FLAC。
5. 音频按数据集放入扁平目录，不保留 tar。
6. 每个数据集生成一个集中存放的 JSONL，并将 `audio_path` 改写为相对路径。
7. 校验 metadata 数量、时长阈值、路径唯一性、文件大小和 FLAC 文件头。
8. 全部完成后生成 `download_summary.json`。

归一化数据中存在不同样本共享同一 tar member 名、甚至同一 tar 内出现同名
member 的情况。脚本使用 metadata 顺序对应 tar 中同名 member 的出现顺序，并将
文件命名为：

```text
<shard>__<member-stem>__<source-path-hash>.flac
```

因此不能直接用 metadata 的 `id` 或 tar member basename 作为扁平文件名。

## 输出结构

```text
s2mel-train-data/
  ears/*.flac
  expresso/*.flac
  Genshin/*.flac
  hi_fi_tts/*.flac
  noiz-short/
  StarRail/*.flac
  vctk/*.flac
  WutheringWaves/*.flac
  metadata/
    ears.jsonl
    expresso.jsonl
    Genshin.jsonl
    hi_fi_tts.jsonl
    noiz-short.jsonl
    StarRail.jsonl
    vctk.jsonl
    WutheringWaves.jsonl
  splits/
    seed1234_valid1000/
      train.jsonl
      valid.jsonl
      split_summary.json
  download_summary.json
```

例如 `metadata/Genshin.jsonl` 中的路径形如：

```json
{"audio_path":"../Genshin/Genshin-000000__sample__0123456789ab.flac"}
```

相对路径以 JSONL 所在的 `metadata/` 为基准。该布局应通过普通
`train_jsonl` loader 使用，不能作为 tar-sharded `train_speechdata_dir` 使用。

## 新机器首次同步

### 1. 准备环境

安装 `uv`、Git 和 tmux，拉取本仓库，然后把 service account key 安全地复制到
新机器。key 路径可以不同，通过 `--key-file` 指定。

建议先确认目标盘至少有约 340 GB 可用空间。脚本按所引用 tar 总大小加 20 GiB
保留空间做保守预检；本次实际输出约 200.26 GB。

```bash
mkdir -p /mnt/data_3t_1/datasets/preprocess/s2mel-train-data
```

`gcloud` 不是必需依赖；脚本通过 `gcsfs` 和 service account key 直接访问 GCS。

### 2. 可选预检

预检会扫描 metadata、检查路径和冲突、统计入选记录，并验证磁盘空间，但不会下载
音频：

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/gcs-key.json"
export GOOGLE_CLOUD_PROJECT="noiz-430406"

uv run --no-project --with gcsfs \
  python scripts/sync_s2mel_train_data.py \
  --key-file "$GOOGLE_APPLICATION_CREDENTIALS" \
  --output-root /mnt/data_3t_1/datasets/preprocess/s2mel-train-data \
  --workers 4 \
  --preflight-only
```

### 3. 在 tmux 中同步

先检查是否已有同名任务，避免重复启动：

```bash
tmux list-sessions
pgrep -af sync_s2mel_train_data.py
```

启动后台同步：

```bash
tmux new-session -d -s s2mel-gcs-duration-gt6 \
  "cd /path/to/semantic2any && \
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-key.json && \
   export GOOGLE_CLOUD_PROJECT=noiz-430406 && \
   set -o pipefail && \
   uv run --no-project --with gcsfs \
     python scripts/sync_s2mel_train_data.py \
       --key-file \$GOOGLE_APPLICATION_CREDENTIALS \
       --output-root /mnt/data_3t_1/datasets/preprocess/s2mel-train-data \
       --min-duration 6 \
       --workers 4 \
     2>&1 | tee /tmp/s2mel-gcs-duration-gt6.log"
```

查看运行状态：

```bash
tmux attach -t s2mel-gcs-duration-gt6
```

在 tmux 中按 `Ctrl-b d` 可退出而不终止任务。Cursor、SSH 或本地网络连接中断不会
影响远端 tmux 中的同步。

## 断点续跑

同一条命令可直接重跑：

- 正式 FLAC 先写入隐藏 `.part` 文件，大小正确后再原子改名。
- 已完整落盘且具有 `fLaC` 文件头的整个 shard 会跳过 GCS 读取。
- 部分完成的 shard 会重新流式读取，但已匹配文件会复用。
- metadata 先写成 `metadata/<dataset>.jsonl.tmp`，该数据集全部通过校验后再原子
  改名为 `.jsonl`。
- `download_summary.json` 只在所有数据集成功后生成。

不要手工删除 `.jsonl.tmp` 或 `.part` 后直接假定任务完成；应重跑脚本，让其完成
校验和正式文件替换。

## 并发与磁盘

默认 `--workers 4`，即同一数据集最多并发读取 4 个 tar。机械盘上不建议盲目增加
到很高的值；网络带宽充足且目标是 SSD 时可以尝试 8。降低到 1 可串行处理。

脚本按数据集顺序处理，不会同时把八个数据集全部展开到内存。metadata 会在预检时
加载，用于建立 tar member 到输出文件的映射。

## 本次同步结果

完成时间：2026-07-13 12:04:26 UTC。

- 源 metadata：1,300,477 条
- 入选音频：439,305 条
- 总时长：1,256.85 小时
- 输出大小：200,262,683,193 bytes
- `ears`：17,125 条，99.93 小时
- `expresso`：1,014 条，2.23 小时
- `Genshin`：248,669 条，670.56 小时
- `hi_fi_tts`：7,931 条，17.10 小时
- `noiz-short`：0 条；其 97,668 条源记录均不严格大于 6 秒
- `StarRail`：130,381 条，371.66 小时
- `vctk`：1,485 条，3.17 小时
- `WutheringWaves`：32,700 条，92.20 小时

权威机器可直接查看：

```bash
python -m json.tool \
  /mnt/data_3t_1/datasets/preprocess/s2mel-train-data/download_summary.json
```

## 固定训练/验证划分

当前训练使用 seed `1234`，从全部 439,305 条记录中按数据源规模比例固定抽取
1,000 条作为验证集，其余 438,305 条作为训练集。验证记录会从训练 manifest 中
移除，两者没有重叠。

同步完成后，在新机器执行：

```bash
uv run python scripts/split_s2mel_validation.py \
  --metadata-dir /mnt/data_3t_1/datasets/preprocess/s2mel-train-data/metadata \
  --output-dir /mnt/data_3t_1/datasets/preprocess/s2mel-train-data/splits/seed1234_valid1000 \
  --valid-size 1000 \
  --seed 1234
```

脚本按文件名排序读取 `metadata/*.jsonl`，先按各数据源记录数进行比例分配，再用
largest-remainder 补足到精确的 1,000 条，最后用固定 seed 从各 manifest 的记录
位置中抽样。生成的 manifest 会把音频路径改写为当前机器上的绝对路径，因此数据根
目录不同时也应在该机器重新运行脚本，不要直接复制另一台机器生成的 split JSONL。

本次划分结果：

- `Genshin`：训练 248,103 条，验证 566 条
- `StarRail`：训练 130,084 条，验证 297 条
- `WutheringWaves`：训练 32,625 条，验证 75 条
- `ears`：训练 17,086 条，验证 39 条
- `expresso`：训练 1,012 条，验证 2 条
- `hi_fi_tts`：训练 7,913 条，验证 18 条
- `vctk`：训练 1,482 条，验证 3 条
- `noiz-short`：0 条

检查划分摘要：

```bash
python -m json.tool \
  /mnt/data_3t_1/datasets/preprocess/s2mel-train-data/splits/seed1234_valid1000/split_summary.json
```

摘要中的 `train_records` 应为 `438305`，`valid_records` 应为 `1000`。只要同步后
各源 manifest 的内容和顺序不变，上述命令会复现相同的验证样本。

## 训练使用

训练单个数据集时指向对应 JSONL：

```bash
TRAIN_JSONL=/mnt/data_3t_1/datasets/preprocess/s2mel-train-data/metadata/ears.jsonl \
  bash scripts/train_s2mel_random_split.sh
```

加载全部数据集时可传入整个 `metadata/` 目录；loader 会读取其中所有非空
`*.jsonl`。`noiz-short.jsonl` 当前为空，不提供训练样本。

复现当前 8 卡 random-split 训练时，`scripts/train_s2mel_random_split.sh` 默认
读取上述固定划分，也可以显式指定：

```bash
TRAIN_JSONL=/mnt/data_3t_1/datasets/preprocess/s2mel-train-data/splits/seed1234_valid1000/train.jsonl \
VALID_JSONL=/mnt/data_3t_1/datasets/preprocess/s2mel-train-data/splits/seed1234_valid1000/valid.jsonl \
NUM_PROCESSES=8 \
  bash scripts/train_s2mel_random_split.sh
```

对应配置为
`configs/s2mel_zipformer_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml`；
当前每 1,000 step 验证一次。
