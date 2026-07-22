# MaskGCT semantic code 预提取

`scripts/precompute_maskgct_codes.py` 将每条音频的 MaskGCT semantic index
保存为连续的 little-endian `uint16` 分片。训练时只加载约 32 MiB 的冻结
MaskGCT lookup table，将 index 还原为原有的 1024 维连续 feature；不会加载
W2V-BERT 或完整 RepCodec encoder。

这里的 lookup table 来自 MaskGCT RepCodec 的 FVQ codebook 和
`out_project`，不是 s2mel length regulator 中可训练的 `embedding`。

音频文件由 DataLoader worker 在 CPU 解码并重采样至 16 kHz，随后以 NumPy
waveform 交给 MaskGCT。这样避免 GPU 重采样后为适配特征提取器再次同步回 CPU。

## 输出结构

```text
<output-dir>/
  semantic_code_metadata.json
  maskgct_lookup.pt
  codes/
    codes.shard00000of00008.bin
  manifests/
    manifest.shard00000of00008.jsonl
  errors/
    errors.shard00000of00008.jsonl
```

训练时将 `--train-jsonl` 或 `--valid-jsonl` 指向 `manifests/` 目录。每条
manifest 记录携带 binary offset、code length、codec fingerprint 和 lookup
checksum；loader 会拒绝混用不匹配的 codebook。

## 单卡小规模验证

先用少量 JSONL 验证资产与输出：

```bash
uv run python scripts/precompute_maskgct_codes.py \
  --config configs/s2mel_zipformer_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml \
  --model-dir checkpoints/feature-extractors \
  --source /path/to/small.jsonl \
  --output-dir /mnt/data_3t_1/datasets/preprocess/maskgct-codes/smoke \
  --device cuda:0 \
  --batch-size 16 \
  --num-workers 8
```

同一命令可以直接重跑。脚本会校验已提交 manifest，截断 binary 中可能存在的
未提交尾部，并只处理尚未完成的记录。默认遇到坏音频即失败；需要记录并跳过时
添加 `--skip-audio-errors`。

## filtered 数据集的 8 卡提取

特征提取本身不负责划分 train/valid。直接读取 `metadata/` 中全部 JSONL，
对 filtered 数据集的 1,413,343 条音频各提取一次；`--num-shards` 仅表示并行
存储分片，不是训练集划分。

当前任务输出到
`/mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes`。

```bash
tmux new-session -d -s maskgct-codes-filtered-8gpu \
  "cd /mnt/data_sdd/hhy/noiz-tts/semantic2any && \
   mkdir -p /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes/logs && \
   for shard in \$(seq 0 7); do \
     CUDA_VISIBLE_DEVICES=\$shard uv run python scripts/precompute_maskgct_codes.py \
       --config configs/s2mel_zipformer_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml \
       --model-dir checkpoints/feature-extractors \
       --source /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/metadata \
       --output-dir /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes \
       --device cuda:0 --batch-size 8 --num-workers 2 \
       --num-shards 8 --shard \$shard \
       > /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes/logs/shard-\$shard.log 2>&1 & \
   done; wait"
```

查看任务：

```bash
tmux attach -t maskgct-codes-filtered-8gpu
```

## 生成固定训练/验证划分

8 个 shard 全部完成后，再从 code manifest 固定抽取 1,000 条验证记录。不要在提取
尚未完成时生成 split，否则后续完成的记录不会进入训练集。

```bash
uv run python scripts/split_s2mel_validation.py \
  --metadata-dir /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes/manifests \
  --output-dir /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes/splits/seed1234_valid1000 \
  --valid-size 1000 \
  --seed 1234
```

split 脚本会保留 semantic offset、length、fingerprint 和 checksum，并将
`audio_path`、`semantic_code_path`、`semantic_lookup_path` 解析为绝对路径。
生成的 train/valid manifest 互不重叠。

## 默认训练规则

默认入口改为：

```bash
NUM_PROCESSES=8 bash scripts/train_s2mel_random_split.sh
```

虽然脚本名为历史遗留的 `random_split`，默认配置已经切换到
`configs/s2mel_zipformer_s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x.yaml`，
并读取上述 code-aware split。

训练样本按 `speaker_id` 组织：

1. 同一说话人有两条及以上可用音频时，target 取 3–30 秒的完整音频，不做裁切；
   超过 30 秒的音频不作为 target，但仍可作为 prompt。
2. prompt 随机选择该说话人的另一条音频，至少 3 秒；超过 20 秒时只保留开头
   20 秒，semantic code 同比例保留对应前缀。
3. 说话人只有一条可用音频时，从该音频随机切分 prompt/target，二者均至少
   3 秒且 prompt 不超过 20 秒。使用预计算 code 时先选择整数 semantic frame
   分界，再映射回音频 sample，因此 code 不会在 frame 中间切开。
4. 验证集不会从训练集借用同说话人 prompt，避免 train/valid 交叉。

当前已生成 code 的 `semantic_max_audio_seconds` 是 30 秒，所以默认配置必须保持
`data.max_audio_seconds: 30.0`。`data.preload_features` 必须保持 `false`；训练只
加载约 32 MiB lookup table，不会重新运行 MaskGCT encoder。

