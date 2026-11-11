"""Bounded efficient SDPA with stable math fallback.

The conservative Cauchy-Schwarz bound applies to every unmasked QK/sqrt(d)
score in this call. 128 is an empirically validated routing threshold, not
an assertion that all inputs below it have exact gradients. No scores,
probabilities, model parameters or gradients are clipped or normalized.
"""
import math
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from .stable_attention import stable_math_attention

SCORE_BOUND_LIMIT = 128.0
FULL_MATH_BOUND_LIMIT = float(2**23)


class AttentionRangeExceeded(RuntimeError):
    pass

COUNTERS = {'fast_calls': 0, 'math_calls': 0, 'max_score_bound': 0.0, 'full_math_retries': 0}


def guarded_attention(q, k, v, mask=None, query_chunk=256, force_math=False, allow_retry=False):
    if force_math:
        COUNTERS['math_calls'] += 1
        return stable_math_attention(q, k, v, mask, query_chunk)
    # Floating masks may add arbitrary bias, which the QK bound does not cover.
    eligible = (q.is_cuda and q.dtype == torch.bfloat16 and
                k.dtype == q.dtype and v.dtype == q.dtype and
                (mask is None or mask.dtype == torch.bool))
    bound = float('inf')
    if eligible:
        with torch.no_grad():
            qnorm = q.float().norm(dim=-1).amax(dim=-1)
            knorm = k.float().norm(dim=-1).amax(dim=-1)
            bound = float((qnorm * knorm).amax()) / math.sqrt(q.shape[-1])
    if math.isfinite(bound):
        COUNTERS['max_score_bound'] = max(COUNTERS['max_score_bound'], bound)
    if allow_retry and (not math.isfinite(bound) or bound > FULL_MATH_BOUND_LIMIT) and eligible:
        raise AttentionRangeExceeded('extreme score bound requires a complete math forward')
    if eligible and math.isfinite(bound) and bound <= SCORE_BOUND_LIMIT:
        COUNTERS['fast_calls'] += 1
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    COUNTERS['math_calls'] += 1
    return stable_math_attention(q, k, v, mask, query_chunk)
