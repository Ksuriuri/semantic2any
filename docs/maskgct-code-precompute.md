# MaskGCT semantic code 预提取

`scripts/precompute_maskgct_codes.py` 将每条音频的 MaskGCT semantic index
保存为连续的 little-endian `uint16` 分片。训练时只加载约 32 MiB 的冻结
MaskGCT lookup table，将 index 还原为原有的 1024 维连续 feature；不会加载
W2V-BERT 或完整 RepCodec encoder。

这里的 lookup table 来自 MaskGCT RepCodec 的 FVQ codebook 和
`out_project`，不是 s2mel length regulator 中可训练的 `embedding`。

音频文件仍由 DataLoader worker 在 CPU 解码，但不会在 worker 中重采样。
同采样率音频会在主进程组成 batch，传到对应 GPU 后统一重采样至 16 kHz。

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

```bash
tmux new-session -d -s maskgct-codes-filtered-8gpu \
  "cd /mnt/data_sdd/hhy/noiz-tts/semantic2any && \
   mkdir -p /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes/logs && \
   for shard in \$(seq 0 7); do \
     CUDA_VISIBLE_DEVICES=\$shard uv run python scripts/precompute_maskgct_codes.py \
       --config configs/s2mel_zipformer_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml \
       --source /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/metadata \
       --output-dir /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes \
       --device cuda:0 --batch-size 16 --num-workers 2 \
       --num-shards 8 --shard \$shard \
       > /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes/logs/shard-\$shard.log 2>&1 & \
   done; wait"
```

查看任务：

```bash
tmux attach -t maskgct-codes-filtered-8gpu
```

## 使用 code manifest 训练

```bash
uv run accelerate launch trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer_s2mel_train_data_random_split_bigvgan_v2_44khz_128band_512x.yaml \
  --train-jsonl /mnt/data_3t_1/datasets/preprocess/s2mel-train-data-filtered/maskgct-codes/manifests
```

上述命令将全部提取记录作为一个数据源使用；若需要 train/valid，应在特征提取
完成后基于生成的 manifest 单独划分，而不是重复提取音频。

`random_split_audio: true` 时，训练仍随机选择实际音频切点，然后按
`split_sample / clipped_audio_samples` 的比例切分整条 semantic code。prompt
和 target 在 length regulator 中仍分别插值，但不会为每个随机切分重新运行
MaskGCT。`data.preload_features` 必须保持 `false`。

