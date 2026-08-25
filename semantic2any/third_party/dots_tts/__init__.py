"""Vendored dots.tts AudioVAE (encoder + BigVGAN-style decoder)."""

from .bigvgan import AudioVAE
from .config import DEFAULT_DOTS_AUDIO_VAE_CONFIG, AudioVAEConfig

__all__ = ["AudioVAE", "AudioVAEConfig", "DEFAULT_DOTS_AUDIO_VAE_CONFIG"]
