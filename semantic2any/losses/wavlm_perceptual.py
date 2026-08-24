"""Frozen WavLM-Large feature L1, used as a perceptual aux on vocoded audio.

Full-reference: GT waveform vs x1_hat -> frozen BigVGAN.  WavLM stays frozen;
only the predicted waveform is in the graph.  Input is resampled to 16 kHz and
per-sequence normalized the way the Wav2Vec2/WavLM feature extractor expects.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class WavLMPerceptualLoss(nn.Module):
    def __init__(
        self,
        model_id: str = "microsoft/wavlm-large",
        cache_dir: str = "",
        layers: tuple[int, ...] = (6, 8, 10, 12),
        input_sr: int = 44100,
        wavlm_sr: int = 16000,
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import WavLMModel

        kwargs: dict = {"local_files_only": bool(local_files_only)}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        model = WavLMModel.from_pretrained(model_id, **kwargs)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.layers = tuple(int(x) for x in layers)
        n_hidden = int(model.config.num_hidden_layers)
        for layer in self.layers:
            if layer < 1 or layer > n_hidden:
                raise ValueError(
                    f"WavLM layer {layer} out of range 1..{n_hidden}"
                )
        self.input_sr = int(input_sr)
        self.wavlm_sr = int(wavlm_sr)

    def _prep(self, wav: torch.Tensor) -> torch.Tensor:
        if self.input_sr != self.wavlm_sr:
            wav = torchaudio.functional.resample(wav, self.input_sr, self.wavlm_sr)
        wav = wav - wav.mean(dim=-1, keepdim=True)
        wav = wav / wav.std(dim=-1, keepdim=True).clamp_min(1e-5)
        return wav

    def _feats(self, wav: torch.Tensor) -> list[torch.Tensor]:
        out = self.model(wav, output_hidden_states=True, return_dict=True)
        # hidden_states[0] is the CNN encoder; [i] is transformer layer i.
        return [out.hidden_states[i] for i in self.layers]

    def forward(
        self,
        pred_wav: torch.Tensor,
        gt_wav: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = self._prep(pred_wav.float())
        with torch.no_grad():
            gt = self._prep(gt_wav.float().detach())
            gt_feats = self._feats(gt)
        pred_feats = self._feats(pred)
        per = []
        for p, g in zip(pred_feats, gt_feats):
            err = (p - g).abs().mean(dim=(1, 2))
            per.append(err)
        stacked = torch.stack(per, dim=0).mean(dim=0)
        if sample_weights is None:
            return stacked.mean()
        w = sample_weights.to(device=stacked.device, dtype=stacked.dtype).reshape(-1)
        return (stacked * w).sum() / w.sum().clamp_min(1e-8)
