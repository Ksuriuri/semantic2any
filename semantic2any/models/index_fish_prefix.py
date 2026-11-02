"""C: reference prefix + ordered sample-clock Index slots, no cross attention."""
import math
import torch
from torch import nn
from .index_fish_mtp import IndexFishMTP, FISH_HZ, valid_mask, time_encoding

INDEX_HOP = 1764  # 44100 / 25, exact integer clock
FISH_HOP = 2048


def index_overlap_slots(frames, source_lengths, *, lookahead_seconds=.08):
    """Return [T,3] source indices and [B,T,3] validity/overlap metadata.

    Each slot is independent: no averaging IDs or embeddings. Source padding
    is excluded. Right-edge overhang is not squeezed into the final frame.
    """
    if frames <= 0 or torch.any(source_lengths <= 0):
        raise ValueError('Positive frame counts required')
    device = source_lengths.device
    start = torch.arange(frames, device=device, dtype=torch.long) * FISH_HOP
    index = start[:, None] // INDEX_HOP + torch.arange(3, device=device)[None]
    left, right = index * INDEX_HOP, (index + 1) * INDEX_HOP
    overlap = (torch.minimum(right, start[:, None] + FISH_HOP) - torch.maximum(left, start[:, None])).clamp_min(0)
    valid = (overlap[None] > 0) & (index[None] < source_lengths[:, None, None])
    if lookahead_seconds is not None:
        deadline = start[:, None] + FISH_HOP + math.floor(lookahead_seconds * 44100 + 1e-7)
        valid = valid & (right[None] <= deadline[None])
    phase = ((left + right).float() * .5 - (start[:, None] + FISH_HOP * .5)) / FISH_HOP
    metadata = torch.stack((phase, overlap.float() / FISH_HOP), -1)
    return index, valid, metadata


def validate_clock_coverage(source_lengths, target_lengths):
    """Every valid Index interval must overlap at least one target interval.

    Reject rather than silently lose extra tokens or stretch clocks. Final
    Index padding may extend beyond the last Fish interval if its start still
    overlaps. Existing manifest duration checks additionally bound short tails.
    """
    if torch.any((source_lengths - 1) * INDEX_HOP >= target_lengths * FISH_HOP):
        raise ValueError('Index tail lies outside Fish time range; inspect paired frame counts')


class IndexFishPrefixMTP(IndexFishMTP):
    def __init__(self, **kwargs):
        kwargs.setdefault('source_mode', 'raw')
        kwargs.setdefault('lookahead_seconds', .08)
        if kwargs['source_mode'] != 'raw':
            raise ValueError('Prefix C requires raw 25 Hz Index codes')
        lookahead = kwargs['lookahead_seconds']
        if lookahead is not None and (not math.isfinite(lookahead) or lookahead < .04):
            raise ValueError('Complete overlap slots require at least 40 ms code lookahead')
        super().__init__(**kwargs)
        for name in ('source_attention', 'reference_attention', 'cross_norm', 'offset'):
            delattr(self, name)
        self.slot_projection = nn.Linear(3 * (self.dim + 3), self.dim)
        self.local_norm = nn.LayerNorm(self.dim)
        self.history_norm = nn.LayerNorm(self.dim)
        self.input_projection = nn.Linear(2 * self.dim, self.dim)
        self.segment_embedding = nn.Embedding(2, self.dim)
        self.separator = nn.Parameter(torch.zeros(self.dim))

    @staticmethod
    def validate_alignment_lengths(source_lengths, target_lengths):
        validate_clock_coverage(source_lengths, target_lengths)

    def aligned_index(self, source_ids, source_lengths, frames):
        idx, valid, meta = index_overlap_slots(frames, source_lengths, lookahead_seconds=self.lookahead_seconds)
        safe = idx.clamp_max(source_ids.shape[1] - 1)
        ids = source_ids[:, safe].masked_fill(~valid, 0)
        emb = self.index_embedding(ids) * valid[..., None]
        metadata = meta[None].expand(source_ids.shape[0], -1, -1, -1) * valid[..., None]
        packed = torch.cat((emb, metadata.to(emb.dtype), valid[..., None].to(emb.dtype)), -1)
        return self.slot_projection(packed.flatten(-2))

    def forward(self, source, source_ids, source_lengths, reference_codes,
                reference_lengths, target_codes, target_lengths):
        b, t, _ = target_codes.shape
        r = reference_codes.shape[1]
        if source_ids.shape != source.shape[:2] or source_ids.shape[0] != b:
            raise ValueError('Source shape mismatch')
        if (torch.any(source_lengths <= 0) or torch.any(source_lengths > source_ids.shape[1])
            or torch.any(reference_lengths <= 0) or torch.any(reference_lengths > r)
            or torch.any(target_lengths <= 0) or torch.any(target_lengths > t)):
            raise ValueError('Lengths outside tensor bounds')
        tv, rv = valid_mask(target_lengths, t), valid_mask(reference_lengths, r)
        codes = target_codes.masked_fill(~tv[..., None], 0)
        ref = reference_codes.masked_fill(~rv[..., None], 0)
        previous = torch.cat((self.bos.expand(b, 1, -1), self.embed_codes(codes[:, :-1])), 1)
        local = self.aligned_index(source_ids, source_lengths, t)
        target = self.input_projection(torch.cat((self.history_norm(previous), self.local_norm(local)), -1))
        target = target + time_encoding((torch.arange(t, device=target.device).float() + .5) / FISH_HZ, self.dim)[None] + self.segment_embedding.weight[1]
        prefix = self.embed_codes(ref) + time_encoding((torch.arange(r, device=target.device).float() + .5) / FISH_HZ, self.dim)[None] + self.segment_embedding.weight[0]
        x = torch.cat((prefix, self.separator.expand(b, 1, -1), target), 1)
        p, n = r + 1, r + 1 + t
        keys = torch.cat((rv, torch.ones(b, 1, device=x.device, dtype=torch.bool), tv), 1)
        allowed = torch.zeros(n, n, device=x.device, dtype=torch.bool)
        allowed[:p, :p] = True  # known reference, bidirectional; never read target
        allowed[p:, :p] = True
        allowed[p:, p:] = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        mask = x.new_zeros(b, n, n).masked_fill(~(allowed[None] & keys[:, None]), float('-inf'))
        for layer in self.condition:
            x = layer(x, mask)
        logits = [self.heads[0](x[:, p:])]
        # Only subsequent codebook heads see current-frame preceding books.
        x = torch.cat((x[:, :p], x[:, p:] + self.books[0](codes[..., 0])), 1)
        for layer in self.shared:
            x = layer(x, mask)
        for k, (mix, layer) in enumerate(zip(self.mix, self.residual), start=1):
            known = torch.cat((x.new_zeros(b, p, self.dim), self.books[k-1](codes[..., k-1])), 1)
            x = layer(mix(torch.cat((x, known), -1)), mask)
            logits.append(self.heads[k](x[:, p:]))
        return logits

    @torch.no_grad()
    def generate(self, source, source_ids, source_lengths, reference_codes,
                 reference_lengths, frames, temperature=0.0):
        validate_clock_coverage(source_lengths, torch.full_like(source_lengths, frames))
        return super().generate(source, source_ids, source_lengths, reference_codes, reference_lengths, frames, temperature)
