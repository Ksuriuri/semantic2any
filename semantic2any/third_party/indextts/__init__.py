"""Minimal IndexTTS-derived runtime components.

See NOTICE.md for provenance, licenses, and the exact upstream revision.
"""

from .audio import mel_spectrogram
from .campplus import CAMPPlus
from .maskgct import RepCodec

__all__ = ["CAMPPlus", "RepCodec", "mel_spectrogram"]
