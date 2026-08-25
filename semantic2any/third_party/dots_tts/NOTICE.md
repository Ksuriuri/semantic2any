# Vendored dots.tts AudioVAE

This directory contains a deliberately minimal, inference-only subset derived
from <https://github.com/studio-dots-ai/dots.tts> (Apache-2.0). Only the frozen
48 kHz AudioVAE encoder/decoder is kept so this project can map semantic codes
onto the latent the official decoder expects. The Qwen2.5 LLM, AR flow-matching
head, tokenizer, and training stack are not vendored.

Included functionality:

- `bigvgan.py`: AudioVAE encoder, MI layers, and BigVGAN-style decoder.
- `alias_free_*.py`: anti-aliased Snake activations used by the decoder.
- `layers.py`: causal Conv1d / ConvTranspose1d used by the decoder.
- `config.py`: AudioVAE hyperparameters without the upstream pydantic base.

Excluded functionality includes the 2B AR TTS backbone, speaker x-vector
encoder, datasets, training scripts, and all model weights.

Copyright and licenses:

- dots.tts: Copyright dots.tts Team / studio-dots-ai, Apache License 2.0.
  The license text is reproduced in `LICENSE`.
- BigVGAN decoder portions: Copyright 2022 NVIDIA CORPORATION, MIT License.
- alias-free-torch activations: Copyright junjun3518, Apache License 2.0.

Apache-2.0: <https://www.apache.org/licenses/LICENSE-2.0>

MIT: <https://opensource.org/license/mit>
