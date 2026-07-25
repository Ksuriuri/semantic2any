# 评估工具

本目录包含 s2mel 模型的完整评估流程，涵盖参考集准备、推理和指标计算。

## 文件说明

| 文件 | 用途 |
|------|------|
| `prepare_vctk_refs.py` | 从 VCTK manifest 准备评估参考集（输入音频 + 截掉 prompt 后的参考尾段） |
| `run_eval.sh` | 针对单个 checkpoint 运行推理 + 指标计算的完整流程 |
| `paired_metrics.py` | 计算 SI-SDR 和 LSD（不需要 GPU 或外部依赖） |
| `summarize_eval.py` | 将多个 run 的指标 JSON 汇总为一张 TSV/JSON 对比表 |

## 依赖

### 基础指标（无额外依赖）

`paired_metrics.py` 只依赖项目本身的 `.venv`（librosa、soundfile、scipy），任意机器均可运行。

### 高级指标（可选，需要 vae-eval 项目）

`run_eval.sh` 的 AudioLDM 指标（FAD、FD、IS、KL）和 SEED-TTS 说话人相似度需要单独的 vae-eval 项目。
将其路径设置为 `VAE` 环境变量即可；若不设置，脚本仅运行 SI-SDR 和 LSD。

vae-eval 使用的外部工具：

- **AudioLDM Evaluation**  
  源码：https://github.com/haoheliu/audioldm_eval  
  依赖已通过 `$VAE/.venv-bigvgan` 安装。

- **SEED-TTS Evaluation**（说话人相似度，WavLM Finetune）  
  源码：https://github.com/BytedanceSpeech/seed-tts-eval  
  权重路径：`$VAE/checkpoints/seed_tts/wavlm_large_finetune.pth`  
  按照该仓库 README 中的链接下载约 1.3 GiB 的 `wavlm_large_finetune.pth`，放置在上述路径。

## 快速开始

### 步骤 1：准备参考集（每个数据集只需运行一次）

```bash
cd /path/to/semantic2any
uv run python eval/prepare_vctk_refs.py \
  --manifest /path/to/vctk_valid.jsonl \
  --full-ref-dir /path/to/vctk_groundtruth_wavs \
  --out-root outputs/eval_refs/vctk10pct_min6_prompt3s \
  --min-duration 6.0 \
  --prompt-seconds 3.01
```

输出目录结构：
```
outputs/eval_refs/vctk10pct_min6_prompt3s/
  input_full_wav/              ← 推理输入（符号链接）
  reference_tail_style-none/   ← 截去 prompt 后的参考尾段
  reference_tail_style-reference/
  manifest_eval.jsonl
  prepare_summary.json
```

### 步骤 2：运行单个 checkpoint 的完整评估

```bash
RUN_NAME=exp06_step34000_prompt3_cfg0_steps25 \
CHECKPOINT=exp/s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x/s2mel_step34000.pth \
CONFIG=configs/s2mel_zipformer_s2mel_train_data_filtered_speaker_pair_bigvgan_v2_44khz_128band_512x.yaml \
VOCODER=checkpoints/vocoders/bigvgan_v2_44khz_128band_512x \
INPUT_REF=outputs/eval_refs/vctk10pct_min6_prompt3s \
GPU=0 \
bash eval/run_eval.sh
```

启用高级指标（需要 vae-eval）：

```bash
VAE=/path/to/vae-eval \
RUN_NAME=exp06_step34000_prompt3_cfg0_steps25 \
... 同上 ... \
bash eval/run_eval.sh
```

快速冒烟测试（只跑一条）：

```bash
SMOKE=1 RUN_NAME=... bash eval/run_eval.sh
```

仅重跑指标（已有生成音频）：

```bash
MODE=metrics RUN_NAME=... bash eval/run_eval.sh
```

### 步骤 3：汇总多个 run 的指标

```bash
uv run python eval/summarize_eval.py \
  --metric-root metrics/ \
  --out-tsv metrics/summary.tsv \
  --out-json metrics/summary.json
```

输出列：`run`、`pairs`、`si_sdr_mean`、`lsd_mean`、`fad`、`fd`、`is_mean`、`kl_sigmoid`、`kl_softmax`、`speaker_similarity_mean`

## 指标说明

| 指标 | 来源 | 含义 |
|------|------|------|
| SI-SDR | `paired_metrics.py` | 信号失真比（越高越好） |
| LSD | `paired_metrics.py` | 对数谱距离（越低越好） |
| FAD | audioldm_eval | Fréchet 音频距离（越低越好） |
| FD | audioldm_eval | Fréchet 距离（VGGish，越低越好） |
| IS | audioldm_eval | Inception Score（越高越好） |
| KL | audioldm_eval | KL 散度（越低越好） |
| Speaker Sim | seed-tts-eval | WavLM Finetune 余弦相似度（越高越好） |
