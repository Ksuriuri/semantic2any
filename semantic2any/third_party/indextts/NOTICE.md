# Vendored IndexTTS runtime components

This directory contains a deliberately minimal, inference-only subset derived
from <https://github.com/index-tts/index-tts> at commit
`7f39d25aee5f5f0e09ce74247c528604dff26ccd`.

Included functionality:

- `audio.py`: IndexTTS s2mel log-mel frontend.
- `campplus.py`: CAMPPlus speaker encoder, originally from 3D-Speaker.
- `maskgct.py`: MaskGCT RepCodec's ConvNeXt encoder/decoder and FVQ only,
  originally from Amphion. The decoder is retained solely because it is part of
  the published checkpoint state dictionary; acoustic decoding is not exposed.
- `bigvgan.py` and `alias_free.py`: NVIDIA BigVGAN inference architecture and
  pure-PyTorch anti-aliased activation path.

Excluded functionality includes IndexTTS GPT models, text frontends, vocoder
training code, MaskGCT semantic-to-acoustic models, acoustic codecs, datasets,
and all model weights.

Copyright and licenses:

- IndexTTS final code: Copyright IndexTTS contributors / bilibili. The
  repository's model-use license is reproduced in `LICENSE.IndexTTS`.
- CAMPPlus: Copyright 3D-Speaker
  (<https://github.com/modelscope/3D-Speaker>), Apache License 2.0.
- MaskGCT RepCodec portions: Copyright 2024 Amphion, MIT License.
- BigVGAN portions: Copyright 2024 NVIDIA CORPORATION, MIT License; adapted
  portions retain the upstream HiFi-GAN (MIT), alias-free-torch (Apache-2.0),
  Julius (MIT), and Snake (MIT) notices in source and here.

Apache-2.0: <https://www.apache.org/licenses/LICENSE-2.0>

MIT: <https://opensource.org/license/mit>

Any modifications made to the original model in this derivative work are not
endorsed, warranted, or guaranteed by the original right-holder of the original
model, and the original right-holder disclaims all liability related to this
derivative work.
