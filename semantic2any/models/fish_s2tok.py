"""Same-rate semantic-to-summed-tokenizer-feature conditional flow model."""
import torch
from torch import nn
from semantic2any.models.flow_matching import CFM

class FishS2Tok(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.DiT.in_channels != 1024 or config.DiT.style_condition:
            raise ValueError('Fish baseline requires 1024 channels and no external style encoder')
        self.condition = nn.Linear(1024, int(config.DiT.content_dim))
        self.cfm = CFM(config)
        self.style_dim = int(config.style_encoder.dim)

    def pack(self, semantic, target, reference, target_lengths, reference_lengths):
        batch = semantic.shape[0]
        lengths = target_lengths + reference_lengths
        if torch.any(target_lengths <= 0) or torch.any(reference_lengths <= 0):
            raise ValueError('Target/reference lengths must be positive')
        if torch.any(target_lengths > semantic.shape[1]) or torch.any(target_lengths > target.shape[1]) or torch.any(reference_lengths > reference.shape[1]):
            raise ValueError('Length exceeds feature tensor')
        width = int(lengths.max())
        x = target.new_zeros(batch, width, 1024)
        cond = semantic.new_zeros(batch, width, 1024)
        for i in range(batch):
            nr, nt = int(reference_lengths[i]), int(target_lengths[i])
            x[i, :nr] = reference[i, :nr]
            x[i, nr:nr+nt] = target[i, :nt]
            cond[i, nr:nr+nt] = semantic[i, :nt]
        mu = self.condition(cond)
        positions = torch.arange(width, device=x.device)[None]
        mu = mu.masked_fill(((positions < reference_lengths[:, None]) | (positions >= lengths[:, None]))[..., None], 0)
        return x.transpose(1, 2), mu, lengths

    def forward(self, semantic, target, reference, target_lengths, reference_lengths):
        x, mu, lengths = self.pack(semantic, target, reference, target_lengths, reference_lengths)
        style = x.new_zeros(x.shape[0], self.style_dim)
        return self.cfm(x, lengths, reference_lengths, mu, style)[0]

    @torch.inference_mode()
    def generate(self, semantic, reference, steps=25, cfg=0.7, temperature=1.0):
        # CFM inference takes a shared prompt length. Decode variable-length batches one at a time.
        if semantic.shape[0] != 1 or reference.shape[0] != 1 or steps <= 0:
            raise ValueError('Inference requires batch=1 and positive steps')
        nt, nr = semantic.shape[1], reference.shape[1]
        tl = torch.tensor([nt], device=semantic.device); rl = torch.tensor([nr], device=semantic.device)
        _, mu, lengths = self.pack(semantic, torch.zeros_like(semantic), reference, tl, rl)
        out = self.cfm.inference(mu, lengths, reference.transpose(1, 2), mu.new_zeros(1,self.style_dim), None, steps, temperature, cfg)
        return out[:, :, nr:nr+nt].transpose(1, 2)
