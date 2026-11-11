"""Stable math SDPA, query tiled and recomputed to bound activation memory.

No Q/K normalization, changed temperature, truncation, or dropout is introduced.
Each query still attends to every unmasked key. Non-reentrant recomputation
preserves autocast; this operation has no RNG because dropout is always zero.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint


def _math_attention(q, k, v, mask):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)


def stable_math_attention(q, k, v, mask=None, query_chunk=256):
    if query_chunk <= 0:
        raise ValueError('query_chunk must be positive')
    outputs=[]
    for start in range(0, q.shape[-2], query_chunk):
        qi=q[...,start:start+query_chunk,:]
        mi=mask if mask is None or mask.shape[-2]==1 else mask[...,start:start+query_chunk,:]
        if torch.is_grad_enabled() and any(x.requires_grad for x in (qi,k,v)):
            oi=checkpoint(_math_attention,qi,k,v,mi,use_reentrant=False,preserve_rng_state=False)
        else:
            oi=_math_attention(qi,k,v,mi)
        outputs.append(oi)
    return torch.cat(outputs,dim=-2)
