# SAC WhisperVQ attribution

This directory contains an inference-only adaptation of the WhisperVQ semantic
encoder used by [Soul-AILab/SAC](https://github.com/Soul-AILab/SAC), based on
upstream commit `2e2c65c6d1437648347b72aec125d89dfbc87621`.

The original implementation is Copyright 2025 Soul-AILab and includes code
derived from Hugging Face Transformers and OpenAI Whisper. It is distributed
under the Apache License 2.0. The adaptation keeps only the 16-layer quantizing
encoder, average pooling, nearest-code lookup, and semantic codebook embedding
required by the SAC-16k-62_5Hz semantic stream.
