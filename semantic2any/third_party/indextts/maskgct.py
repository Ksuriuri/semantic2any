"""Minimal MaskGCT RepCodec implementation used by IndexTTS.

This keeps the upstream module attribute hierarchy intact for checkpoint
compatibility while omitting MaskGCT's acoustic decoder and training stack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
from transformers import Wav2Vec2BertModel


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x).transpose(1, 2)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        if self.gamma is not None:
            x = self.gamma * x
        return residual + x.transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value=None,
        adanorm_num_embeddings=None,
    ):
        super().__init__()
        if adanorm_num_embeddings is not None:
            raise ValueError("RepCodec does not use adaptive normalization")
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = False
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(dim, intermediate_dim, layer_scale_init_value)
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.norm(self.embed(x).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


def _wn_conv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=0.005,
        codebook_loss_weight=1.0,
        use_l2_normlize=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.use_l2_normlize = use_l2_normlize
        if input_dim != codebook_dim:
            self.in_project = _wn_conv1d(input_dim, codebook_dim, kernel_size=1)
            self.out_project = _wn_conv1d(codebook_dim, input_dim, kernel_size=1)
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight
        if self.use_l2_normlize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange(
            (-dist).max(1)[1], "(b t) -> b t", b=latents.size(0)
        )
        return self.decode_code(indices), indices

    def forward(self, z):
        z_e = self.in_project(z)
        z_q, indices = self.decode_latents(z_e)
        if self.training:
            commit_loss = (
                F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
                * self.commitment
            )
            codebook_loss = (
                F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
                * self.codebook_loss_weight
            )
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)
            codebook_loss = torch.zeros(z.shape[0], device=z.device)
        z_q = self.out_project(z_e + (z_q - z_e).detach())
        return z_q, commit_loss, codebook_loss, indices, z_e

    def vq2emb(self, vq, out_proj=True):
        embedding = self.decode_code(vq)
        return self.out_project(embedding) if out_proj else embedding


class ResidualVQ(nn.Module):
    def __init__(
        self,
        input_dim=256,
        num_quantizers=8,
        codebook_size=1024,
        codebook_dim=256,
        quantizer_type="fvq",
        quantizer_dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        if quantizer_type != "fvq":
            raise ValueError("Vendored RepCodec supports only its FVQ quantizer")
        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_type = quantizer_type
        self.quantizer_dropout = quantizer_dropout
        self.quantizers = nn.ModuleList(
            [
                FactorizedVectorQuantize(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z, n_quantizers=None):
        quantized_out = 0.0
        residual = z
        all_commit_losses = []
        all_codebook_losses = []
        all_indices = []
        all_quantized = []
        n_quantizers = self.num_quantizers if n_quantizers is None else n_quantizers
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.num_quantizers + 1
            dropout = torch.randint(1, self.num_quantizers + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)
        for index, quantizer in enumerate(self.quantizers):
            if not self.training and index >= n_quantizers:
                break
            z_q, commit_loss, codebook_loss, indices, _ = quantizer(residual)
            mask = torch.full((z.shape[0],), index, device=z.device) < n_quantizers
            quantized_out = quantized_out + z_q * mask[:, None, None]
            residual = residual - z_q
            all_commit_losses.append((commit_loss * mask).mean())
            all_codebook_losses.append((codebook_loss * mask).mean())
            all_indices.append(indices)
            all_quantized.append(z_q)
        return (
            quantized_out,
            torch.stack(all_indices),
            torch.stack(all_commit_losses),
            torch.stack(all_codebook_losses),
            torch.stack(all_quantized),
        )

    def vq2emb(self, vq, n_quantizers=None):
        quantized_out = 0.0
        n_quantizers = self.num_quantizers if n_quantizers is None else n_quantizers
        for index, quantizer in enumerate(self.quantizers):
            if index >= n_quantizers:
                break
            quantized_out += quantizer.vq2emb(vq[index])
        return quantized_out


class RepCodec(nn.Module):
    def __init__(
        self,
        codebook_size=8192,
        hidden_size=1024,
        codebook_dim=8,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        num_quantizers=1,
        downsample_scale=1,
        cfg=None,
    ):
        super().__init__()
        values = {
            "codebook_size": codebook_size,
            "hidden_size": hidden_size,
            "codebook_dim": codebook_dim,
            "vocos_dim": vocos_dim,
            "vocos_intermediate_dim": vocos_intermediate_dim,
            "vocos_num_layers": vocos_num_layers,
            "num_quantizers": num_quantizers,
            "downsample_scale": downsample_scale,
        }
        if cfg is not None:
            for name in values:
                if hasattr(cfg, name):
                    values[name] = getattr(cfg, name)
        for name, value in values.items():
            setattr(self, name, value)
        if self.downsample_scale is not None and self.downsample_scale > 1:
            self.down = nn.Conv1d(
                self.hidden_size, self.hidden_size, kernel_size=3, stride=2, padding=1
            )
            self.up = nn.Conv1d(
                self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1
            )
        self.encoder = nn.Sequential(
            VocosBackbone(
                self.hidden_size,
                self.vocos_dim,
                self.vocos_intermediate_dim,
                self.vocos_num_layers,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                self.hidden_size,
                self.vocos_dim,
                self.vocos_intermediate_dim,
                self.vocos_num_layers,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )
        self.quantizer = ResidualVQ(
            input_dim=self.hidden_size,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            codebook_dim=self.codebook_dim,
            quantizer_type="fvq",
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normlize=True,
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def quantize(self, x):
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = F.gelu(self.down(x.transpose(1, 2))).transpose(1, 2)
        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        quantized_out, all_indices, _, _, _ = self.quantizer(x)
        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)


def build_semantic_model(stat_path, *, model_path):
    model = Wav2Vec2BertModel.from_pretrained(model_path)
    model.eval()
    stats = torch.load(stat_path, map_location="cpu")
    return model, stats["mean"], torch.sqrt(stats["var"])


def build_semantic_codec(cfg):
    return RepCodec(cfg=cfg).eval()
