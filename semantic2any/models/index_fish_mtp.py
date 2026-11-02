"""Index-conditioned, time-causal / codebook-autoregressive Fish predictor.

Inspired by Drama3's shared enrichment and residual codebook tower. This is
an independent trainable implementation, not checkpoint-compatible with it.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

FISH_HZ = 44100 / 2048

def valid_mask(lengths, width):
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]

def time_encoding(times, dim):
    freq = torch.exp(torch.arange(0, dim, 2, device=times.device).float() * (-math.log(10000) / dim))
    phase = times[..., None] * 50 * freq
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
    def forward(self, x, memory, bias):
        b, t, d = x.shape
        def split(y):
            return y.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(memory)), split(self.v(memory)), attn_mask=bias[:, None].to(x.dtype))
        return self.o(y.transpose(1, 2).reshape(b, t, d))

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def forward(self, x, mask):
        h = self.norm(x)
        x = x + self.attn(h, h, mask)
        return x + self.ff(x)

class IndexFishMTP(nn.Module):
    def __init__(self, dim=256, heads=8, condition_layers=2, shared_layers=2,
                 index_dim=1024, index_vocab=8192, time_sigma=0.15,
                 source_mode="decoded", lookahead_seconds=None,
                 codebook_sizes=(4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)):
        super().__init__()
        if dim % heads or dim % 2 or time_sigma <= 0 or len(codebook_sizes) != 10:
            raise ValueError('Invalid dimensions, time prior, or codebooks')
        if source_mode not in ('raw', 'decoded') or (lookahead_seconds is not None and (source_mode != 'raw' or lookahead_seconds < 0)):
            raise ValueError('Bounded lookahead requires raw codes; decoded features contain future context')
        self.source_mode, self.lookahead_seconds = source_mode, lookahead_seconds
        self.source_hz = 25 if source_mode == 'raw' else 50
        self.dim, self.time_sigma = dim, time_sigma
        self.sizes = tuple(codebook_sizes)
        self.index_projection = nn.Linear(index_dim, dim) if source_mode == "decoded" else None
        # Lossless ID access in addition to frozen decoder features. Never average
        # neighboring Index tokens; repeat only attaches an ID to its two frames.
        self.index_embedding = nn.Embedding(index_vocab, dim)
        self.books = nn.ModuleList(nn.Embedding(n, dim) for n in self.sizes)
        self.bos = nn.Parameter(torch.zeros(dim))
        self.cross_norm = nn.LayerNorm(dim)
        self.source_attention = Attention(dim, heads)
        self.reference_attention = Attention(dim, heads)
        self.condition = nn.ModuleList(Block(dim, heads) for _ in range(condition_layers))
        self.shared = nn.ModuleList(Block(dim, heads) for _ in range(shared_layers))
        self.mix = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(9))
        self.residual = nn.ModuleList(Block(dim, heads) for _ in range(9))
        self.heads = nn.ModuleList(nn.Linear(dim, n) for n in self.sizes)
        self.offset = nn.Parameter(torch.zeros(()))

    def embed_codes(self, codes):
        return sum(book(codes[..., k]) for k, book in enumerate(self.books)) / math.sqrt(10)

    def forward(self, source, source_ids, source_lengths, reference_codes,
                reference_lengths, target_codes, target_lengths):
        b, t, _ = target_codes.shape
        s = source.shape[1]
        if source_ids.shape != source.shape[:2] or torch.any(source_lengths <= 0) or torch.any(reference_lengths <= 0):
            raise ValueError('Missing source/reference frames')
        if torch.any(target_lengths <= 0) or torch.any(target_lengths > t) or torch.any(source_lengths > s) or torch.any(reference_lengths > reference_codes.shape[1]):
            raise ValueError('Lengths outside tensor bounds')
        target_valid = valid_mask(target_lengths, t)
        # Padding may contain arbitrary values, never send them to embeddings.
        codes = target_codes.masked_fill(~target_valid[..., None], 0)
        sv = valid_mask(source_lengths, s)
        source = source.masked_fill(~sv[..., None], 0)
        source_ids = source_ids.masked_fill(~sv, 0)
        rv = valid_mask(reference_lengths, reference_codes.shape[1])
        reference_codes = reference_codes.masked_fill(~rv[..., None], 0)
        st = (torch.arange(s, device=source.device).float() + 0.5) / self.source_hz
        tt = (torch.arange(t, device=source.device).float() + 0.5) / FISH_HZ
        memory = self.index_embedding(source_ids) + time_encoding(st, self.dim)[None]
        if self.index_projection is not None:
            memory = memory + self.index_projection(source)
        previous = torch.cat((self.bos.expand(b, 1, -1), self.embed_codes(codes[:, :-1])), 1)
        x = previous + time_encoding(tt, self.dim)[None]
        # A soft, bounded prior, not a hard window: every valid source frame
        # remains available, including both subframes of every Index token.
        delta = tt[:, None] - st[None] + 0.1 * self.offset.tanh()
        prior = (-0.5 * (delta / self.time_sigma).square()).clamp_min(-12)
        source_bias = prior[None].expand(b, -1, -1).masked_fill(~sv[:, None], float('-inf'))
        if self.lookahead_seconds is not None:
            # A token becomes available at the END of its 40ms interval, not its center.
            available = (torch.arange(s, device=source.device).float() + 1) / self.source_hz
            deadline = (torch.arange(t, device=source.device).float() + 1) / FISH_HZ + self.lookahead_seconds
            allowed = available[None, :] <= deadline[:, None] + 1e-6
            source_bias = source_bias.masked_fill(~allowed[None], float('-inf'))
        reference_bias = source.new_zeros(b, t, reference_codes.shape[1]).masked_fill(~rv[:, None], float('-inf'))
        x = x + self.source_attention(self.cross_norm(x), memory, source_bias)
        x = x + self.reference_attention(self.cross_norm(x), self.embed_codes(reference_codes), reference_bias)
        causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()[None] & target_valid[:, None]
        mask = x.new_zeros(b, t, t).masked_fill(~causal, float('-inf'))
        for layer in self.condition:
            x = layer(x, mask)
        logits = [self.heads[0](x)]
        # c0 is known only for heads 1..9; c0 head above cannot see it.
        x = x + self.books[0](codes[..., 0])
        for layer in self.shared:
            x = layer(x, mask)
        for k, (mix, layer) in enumerate(zip(self.mix, self.residual), start=1):
            x = mix(torch.cat((x, self.books[k-1](codes[..., k-1])), -1))
            x = layer(x, mask)
            logits.append(self.heads[k](x))
        return logits

    def loss(self, logits, codes, lengths):
        mask = valid_mask(lengths, codes.shape[1])
        if not mask.any():
            raise ValueError('Empty targets')
        losses = torch.stack([F.cross_entropy(logit[mask].float(), codes[..., k][mask]) for k, logit in enumerate(logits)])
        return losses.mean(), losses

    @torch.no_grad()
    def generate(self, source, source_ids, source_lengths, reference_codes,
                 reference_lengths, frames, temperature=0.0):
        """Correctness-first generation, full prefix recomputation (no KV cache)."""
        if frames <= 0 or temperature < 0:
            raise ValueError('Invalid frames/temperature')
        b = source.shape[0]
        codes = torch.zeros(b, frames, 10, device=source.device, dtype=torch.long)
        for t in range(frames):
            lengths = torch.full((b,), t + 1, device=source.device, dtype=torch.long)
            for k in range(10):
                logits = self(source, source_ids, source_lengths, reference_codes, reference_lengths, codes[:, :t+1], lengths)[k][:, -1].float()
                if temperature == 0:
                    value = logits.argmax(-1)
                else:
                    value = torch.multinomial((logits / temperature).softmax(-1), 1)[:, 0]
                codes[:, t, k] = value
        return codes


class RawCodeStream:
    """Incremental *codes* API, batch one, fixed reference available upfront.

    Uses full-prefix recomputation for correctness, not a real-time speed claim.
    Each push returns only newly completed Fish frames [1,T,10]. No waveform
    streaming/overlap buffering is implied by this class.
    """
    def __init__(self, model, reference_codes, reference_lengths):
        if model.source_mode != 'raw' or model.lookahead_seconds is None:
            raise ValueError('Streaming requires raw codes and bounded lookahead')
        if model.training or reference_codes.shape[0] != 1:
            raise ValueError('Use eval() and batch size one')
        self.model = model
        self.reference, self.reference_lengths = reference_codes, reference_lengths
        self.source = torch.empty(1, 0, dtype=torch.long, device=reference_codes.device)
        self.codes = torch.empty(1, 0, 10, dtype=torch.long, device=reference_codes.device)
        self.closed = False

    @torch.no_grad()
    def push(self, new_codes, *, final=False, duration_seconds=None):
        if self.closed or new_codes.ndim != 1 or new_codes.dtype != torch.long:
            raise ValueError('Closed stream or invalid integer [N] chunk')
        if new_codes.numel() and (new_codes.min() < 0 or new_codes.max() >= self.model.index_embedding.num_embeddings):
            raise ValueError('Index code outside vocabulary')
        self.source = torch.cat((self.source, new_codes[None].to(self.source.device)), 1)
        n = self.source.shape[1]
        if not n:
            raise ValueError('Empty stream')
        duration = n / 25
        if final:
            duration = duration if duration_seconds is None else duration_seconds
            if not math.isfinite(duration) or duration <= 0 or abs(duration - n / 25) > .081:
                raise ValueError('Invalid final duration')
            end = math.ceil(duration * FISH_HZ)
        else:
            if duration_seconds is not None:
                raise ValueError('Duration is only accepted for final chunk')
            end = max(0, math.floor((duration - self.model.lookahead_seconds) * FISH_HZ + 1e-6))
        if final and hasattr(self.model, "validate_alignment_lengths"):
            self.model.validate_alignment_lengths(torch.tensor([n], device=self.source.device), torch.tensor([end], device=self.source.device))
        start = self.codes.shape[1]
        if end < start:
            raise ValueError('Final duration would retract emitted frames')
        source = self.model.bos.new_zeros(1, n, 1)
        sl = torch.tensor([n], device=self.source.device)
        for t in range(start, end):
            self.codes = torch.cat((self.codes, torch.zeros(1, 1, 10, device=self.source.device, dtype=torch.long)), 1)
            tl = torch.tensor([t+1], device=self.source.device)
            for k in range(10):
                logits = self.model(source, self.source, sl, self.reference, self.reference_lengths, self.codes, tl)
                self.codes[:, t, k] = logits[k][:, -1].argmax(-1)
        self.closed = final
        return self.codes[:, start:].clone()


def build_index_fish_model(model_config, architecture="cross_attention"):
    """Explicit architecture selection; legacy checkpoints retain old behavior."""
    if architecture == "cross_attention":
        return IndexFishMTP(**model_config)
    if architecture == "prefix_aligned":
        from .index_fish_prefix import IndexFishPrefixMTP
        return IndexFishPrefixMTP(**model_config)
    raise ValueError(f"Unknown IndexFish architecture: {architecture}")
