# 模型资产与跨机器部署

本项目的训练和推理源码均随仓库提供，不需要另行克隆 IndexTTS、
BigVGAN、SAC 或 MaskGCT 源码。模型权重不纳入 Git；请根据所选语义
编码器和声码器按需下载，避免拉取完整上游仓库。

以下命令均从仓库根目录执行。先安装锁定的 Python 环境：

```bash
uv sync --frozen
```

## 1. 选择训练所需的语义编码器

MaskGCT 与 SAC 二选一。只有进行带风格条件的训练或推理时才需要
CAMPPlus；`style_condition: false` 的实验不需要该权重。

### MaskGCT（默认）

MaskGCT 训练所需的最小资产约 2.36 GiB：

```bash
ASSET_DIR=checkpoints/feature-extractors

uv run hf download facebook/w2v-bert-2.0 \
  config.json preprocessor_config.json model.safetensors \
  --revision da985ba0987f70aaeb84a80f2851cfac8c697a7b \
  --local-dir "$ASSET_DIR/w2v-bert-2.0"

uv run hf download IndexTeam/IndexTTS-2 \
  wav2vec2bert_stats.pt \
  --revision 740dcaff396282ffb241903d150ac011cd4b1ede \
  --local-dir "$ASSET_DIR"

uv run hf download amphion/MaskGCT \
  semantic_codec/model.safetensors \
  --revision 265c6cef07625665d0c28d2faafb1415562379dc \
  --local-dir "$ASSET_DIR"
```

带风格条件时再下载约 27 MiB 的 CAMPPlus：

```bash
uv run hf download funasr/campplus \
  campplus_cn_common.bin \
  --revision e4b6ede7ce16997aff4ae69fbca1f0175e2afede \
  --local-dir checkpoints/feature-extractors/campplus
```

对应下载页：

- [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0)
- [IndexTeam/IndexTTS-2](https://huggingface.co/IndexTeam/IndexTTS-2)
- [amphion/MaskGCT](https://huggingface.co/amphion/MaskGCT)
- [funasr/campplus](https://huggingface.co/funasr/campplus)

不要下载 `IndexTeam/IndexTTS-2` 中的 `gpt.pth`、`s2mel.pth`、
`qwen0.6bemo4-merge/`、BPE 和情感矩阵；本项目的 semantic-to-mel
训练不使用它们。也不要下载 MaskGCT 的 acoustic codec、T2S 或 S2A
权重。

### SAC

SAC 路径只使用 GLM-4-Voice 的语义 tokenizer，约 1.36 GiB；不会下载
`Soul-AILab/SAC-16k-62_5Hz` 的声学编码器、解码器或完整 codec：

```bash
uv run hf download zai-org/glm-4-voice-tokenizer \
  config.json preprocessor_config.json model.safetensors \
  --revision a5f2404e63c84e92f5238908e1706316324ebafa \
  --local-dir checkpoints/sac-tokenizer
```

在配置中设置：

```yaml
paths:
  model_dir: checkpoints/feature-extractors
semantic_codec:
  type: sac
  tokenizer_path: checkpoints/sac-tokenizer
  local_files_only: true
```

若启用风格条件，仍需按上一节下载 CAMPPlus。下载页：
[zai-org/glm-4-voice-tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer)。

## 2. 推理时按 mel 规格选择一个声码器

训练和特征预计算不需要 BigVGAN。仅在运行
`scripts/infer_s2mel_zipformer.py` 时，下载与配置采样率、mel band 和
hop size 完全一致的一个生成器。每个目录只需 `config.json` 和
`bigvgan_generator.pt`，不要下载约 1.4–1.5 GB 的 discriminator/optimizer
文件。

22.05 kHz、80-band、256-hop：

```bash
uv run hf download nvidia/bigvgan_v2_22khz_80band_256x \
  config.json bigvgan_generator.pt \
  --revision 633ff708ed5b74903e86ff1298cf4a98e921c513 \
  --local-dir checkpoints/vocoders/bigvgan_v2_22khz_80band_256x
```

44.1 kHz、128-band、256-hop：

```bash
uv run hf download nvidia/bigvgan_v2_44khz_128band_256x \
  config.json bigvgan_generator.pt \
  --revision 30ba327a3795ba672b9c972bb0587e8b571eb420 \
  --local-dir checkpoints/vocoders/bigvgan_v2_44khz_128band_256x
```

44.1 kHz、128-band、512-hop：

```bash
uv run hf download nvidia/bigvgan_v2_44khz_128band_512x \
  config.json bigvgan_generator.pt \
  --revision 95a9d1dcb12906c03edd938d77b9333d6ded7dfb \
  --local-dir checkpoints/vocoders/bigvgan_v2_44khz_128band_512x
```

将配置中的 `vocoder.model_id` 指向所选本地目录，或在推理时传入
`--vocoder-model`。下载页：

- [22 kHz / 80 band / 256 hop](https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x)
- [44 kHz / 128 band / 256 hop](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_256x)
- [44 kHz / 128 band / 512 hop](https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_512x)

## 3. 启动训练

数据 JSONL、输出目录和资产目录均可在新机器上覆盖，不需要修改源码：

```bash
tmux new-session -s s2mel-train

uv run accelerate launch trainers/train_s2mel_zipformer.py \
  --config configs/s2mel_zipformer.yaml \
  --model-dir checkpoints/feature-extractors \
  --train-jsonl /data/train.jsonl \
  --valid-jsonl /data/valid.jsonl \
  --output-dir /data/experiments/s2mel
```

按 `Ctrl-b d` 脱离，之后使用 `tmux attach -t s2mel-train` 恢复。
项目自身训练产生的 `s2mel_*.pth` 不属于上游预训练资产；推理时需通过
`--checkpoint` 指向实际实验产物。

如果训练和验证 manifest 已同时提供 `mel_path`、`semantic_path` 与
`style_path`，训练器不会初始化在线冻结特征提取器；这类训练可不传输
MaskGCT、SAC 和 CAMPPlus 资产。预计算特征必须携带匹配的 codec
fingerprint，不能在 MaskGCT 与 SAC 实验间混用。

只预提取紧凑 MaskGCT code、训练时通过冻结词表恢复连续 feature 的流程见
[maskgct-code-precompute.md](maskgct-code-precompute.md)。

## 4. 可选评估资产

`scripts/task12_run_semantic2any_eval.sh` 是历史实验评估入口，不参与安装、
训练或常规推理。其 AudioLDM 指标和 SEED-TTS WavLM 说话人相似度需要
额外评估源码及约 1.3 GiB 的 `wavlm_large_finetune.pth`。这些资产不要为
训练机器预下载；只有复现 task12 指标时，才按上游说明准备：

- [AudioLDM Evaluation](https://github.com/haoheliu/audioldm_eval)
- [SEED-TTS Evaluation](https://github.com/BytedanceSpeech/seed-tts-eval)

task12 脚本中的这组依赖属于可选评估工具，不是 semantic2any 核心运行时
依赖。

## 5. 离线迁移与校验

联网机器完成下载后，只需传输仓库、所选资产目录、数据 manifest 和需要
恢复的训练 checkpoint。离线机器应将 `semantic_codec.local_files_only`
和 `vocoder.local_files_only` 设为 `true`。

最小目录应类似：

```text
checkpoints/
  feature-extractors/
    campplus/campplus_cn_common.bin          # 仅风格条件需要
    semantic_codec/model.safetensors        # 仅 MaskGCT
    w2v-bert-2.0/{config.json,preprocessor_config.json,model.safetensors}
    wav2vec2bert_stats.pt                    # 仅 MaskGCT
  sac-tokenizer/                             # 仅 SAC
    {config.json,preprocessor_config.json,model.safetensors}
  vocoders/<所选 BigVGAN>/                   # 仅推理
    {config.json,bigvgan_generator.pt}
```

请同时遵守各下载页的模型许可证；仓库内同步源码的来源和许可证记录在
`semantic2any/third_party/indextts/NOTICE.md`。
