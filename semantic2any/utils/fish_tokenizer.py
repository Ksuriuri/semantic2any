"""Frozen Drama-3 quantizer features. S is BEFORE post_module/upsample."""
from pathlib import Path
import hashlib
import sys
import torch
from torch import nn

MODEL_SHA256 = '74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f'

def load_fish_codec(repo, checkpoint, expected_sha256=MODEL_SHA256):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    path = Path(checkpoint)
    with path.open('rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != expected_sha256:
        raise ValueError('Fish checkpoint SHA256 mismatch')
    sys.path.insert(0, str(Path(repo).resolve()))
    cfg = OmegaConf.load(Path(repo) / 'fish_speech/configs/modded_dac_vq.yaml')
    codec = instantiate(cfg)
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    state = state.get('state_dict', state)
    if any(k.startswith('generator.') for k in state):
        state = {k.removeprefix('generator.'): v for k, v in state.items() if k.startswith('generator.')}
    # Official nonpersistent attention caches are regenerated from the config.
    state = {k: v for k, v in state.items() if not k.endswith(('.freqs_cis', '.causal_mask'))}
    codec.load_state_dict(state, strict=True, assign=True)
    return codec.float().eval().requires_grad_(False)

class FishCodeFeatures(nn.Module):
    """Code lookup + each quantizer's out_proj, frozen, exact FP32 tables."""
    def __init__(self, tables):
        super().__init__()
        if len(tables) != 10 or any(t.ndim != 2 or t.shape[1] != 1024 for t in tables):
            raise ValueError('Expected ten projected 1024-dimensional codebooks')
        self.tables = nn.ModuleList([nn.Embedding.from_pretrained(t.float(), freeze=True) for t in tables])

    @classmethod
    @torch.no_grad()
    def from_codec(cls, codec):
        q = codec.quantizer
        quantizers = [*q.semantic_quantizer.quantizers, *q.quantizer.quantizers]
        tables = []
        for vq in quantizers:
            ids = torch.arange(vq.codebook.weight.shape[0], device=vq.codebook.weight.device)[None]
            tables.append(vq.out_proj(vq.decode_code(ids))[0].T.detach().clone())
        return cls(tables)

    def _check(self, codes):
        if codes.ndim != 3 or codes.shape[-1] != 10 or codes.dtype not in (torch.int64, torch.int32):
            raise ValueError('codes must be integer [B,T,10]')
        for i, table in enumerate(self.tables):
            if torch.any(codes[..., i] < 0) or torch.any(codes[..., i] >= table.num_embeddings):
                raise ValueError(f'Invalid code in codebook {i}')

    @torch.no_grad()
    def forward(self, codes):
        self._check(codes)
        semantic = self.tables[0](codes[..., 0])
        residual = self.tables[1](codes[..., 1])
        for i in range(2, 10):
            residual = residual + self.tables[i](codes[..., i])
        return semantic, semantic + residual

    @torch.no_grad()
    def semantic(self, codes0):
        if codes0.ndim != 2 or codes0.dtype not in (torch.int64, torch.int32):
            raise ValueError('semantic codes must be integer [B,T]')
        if torch.any(codes0 < 0) or torch.any(codes0 >= self.tables[0].num_embeddings):
            raise ValueError('Semantic code out of range')
        return self.tables[0](codes0)

@torch.no_grad()
def decode_summed_features(codec, summed):
    """[B,T,1024] S -> frozen post_module -> upsample -> waveform."""
    if summed.ndim != 3 or summed.shape[-1] != 1024:
        raise ValueError('S must be [B,T,1024]')
    z = codec.quantizer.upsample(codec.quantizer.post_module(summed.transpose(1, 2)))
    return codec.decoder(z)
