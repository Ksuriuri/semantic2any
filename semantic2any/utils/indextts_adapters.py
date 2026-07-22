from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from semantic2any.defaults import (
    DEFAULT_MEL_CHANNELS,
    DEFAULT_MEL_HOP_LENGTH,
    DEFAULT_MEL_N_FFT,
    DEFAULT_MEL_SAMPLE_RATE,
    DEFAULT_MEL_WIN_LENGTH,
)
from semantic2any.data.s2mel_dataset import (
    DEFAULT_MAX_AUDIO_SECONDS,
    DEFAULT_MAX_PAIR_SECONDS,
    DEFAULT_MAX_PROMPT_SECONDS,
    choose_prompt_len,
    collate_paired_features,
)
from semantic2any.third_party.indextts import CAMPPlus, mel_spectrogram


_USE_CONFIGURED_AUDIO_LIMIT = object()


def _get(obj, name: str, default=None):
    return getattr(obj, name, obj.get(name, default) if isinstance(obj, dict) else default)


def _uses_style_condition(cfg: Any) -> bool:
    s2mel_cfg = _get(cfg, "s2mel")
    dit_type = str(_get(s2mel_cfg, "dit_type", "ZipFormer")).lower()
    estimator_cfg = _get(s2mel_cfg, "DiT" if dit_type == "dit" else "ZipFormer")
    return bool(_get(estimator_cfg, "style_condition", False))


def _resolve_model_path(model_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = model_dir / path
    return path


def _load_audio(path: str | Path, max_audio_seconds: float | None = None) -> tuple[torch.Tensor, int]:
    audio, sr = torchaudio.load(str(path))
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if max_audio_seconds is not None:
        max_samples = int(max_audio_seconds * sr)
        audio = audio[:, :max_samples]
    return audio, sr


class S2MelFeatureAdapter(nn.Module):
    """Frozen mel/style stack with a selectable semantic codec backend."""

    def __init__(
        self,
        cfg: Any,
        *,
        semantic_lookup_path: str | Path | None = None,
        semantic_lookup_sha256: str | None = None,
    ) -> None:
        super().__init__()

        paths_cfg = _get(cfg, "paths")
        style_cfg = _get(_get(cfg, "s2mel"), "style_encoder")
        self.style_dim = int(_get(style_cfg, "dim", 192))
        self.use_style_condition = _uses_style_condition(cfg)
        self.model_dir = Path(_get(paths_cfg, "model_dir")).expanduser().resolve()

        from semantic2any.utils.semantic_codecs import (
            MaskGCTCodebookDecoder,
            build_semantic_codec,
            semantic_codec_type,
        )

        needs_model_dir = (
            (semantic_lookup_path is None and semantic_codec_type(cfg) == "maskgct")
            or self.use_style_condition
        )
        if needs_model_dir and not self.model_dir.exists():
            raise FileNotFoundError(f"model_dir does not exist: {self.model_dir}")
        self.semantic_decoder: MaskGCTCodebookDecoder | None = None
        self.semantic_backend: nn.Module | None = None
        if semantic_lookup_path is not None:
            if semantic_codec_type(cfg) != "maskgct":
                raise ValueError("Precomputed MaskGCT codes require semantic_codec.type=maskgct")
            self.semantic_decoder = MaskGCTCodebookDecoder(
                semantic_lookup_path,
                expected_sha256=semantic_lookup_sha256,
            )
        else:
            self.semantic_backend = build_semantic_codec(cfg, model_dir=self.model_dir)

        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

        self.campplus_model: CAMPPlus | None = None
        if self.use_style_condition:
            campplus_ckpt = _resolve_model_path(
                self.model_dir,
                _get(paths_cfg, "campplus_ckpt", "campplus/campplus_cn_common.bin"),
            )
            self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=self.style_dim)
            self.campplus_model.load_state_dict(
                torch.load(campplus_ckpt, map_location="cpu"), strict=True
            )
            self.campplus_model.eval()

        preprocess = _get(cfg, "preprocess_params", _get(_get(cfg, "s2mel"), "preprocess_params", None))
        spect = _get(preprocess, "spect_params")
        fmax = _get(spect, "fmax", "None")
        fmax = None if fmax in (None, "None") else float(fmax)
        self.mel_args = {
            "n_fft": int(_get(spect, "n_fft", DEFAULT_MEL_N_FFT)),
            "num_mels": int(_get(spect, "n_mels", DEFAULT_MEL_CHANNELS)),
            "sampling_rate": int(
                _get(preprocess, "sr", DEFAULT_MEL_SAMPLE_RATE)
            ),
            "hop_size": int(_get(spect, "hop_length", DEFAULT_MEL_HOP_LENGTH)),
            "win_size": int(
                _get(spect, "win_length", DEFAULT_MEL_WIN_LENGTH)
            ),
            "fmin": float(_get(spect, "fmin", 0)),
            "fmax": fmax,
            "center": False,
        }
        self.mel_spectrogram = mel_spectrogram

        data_cfg = _get(cfg, "data")
        self.max_audio_seconds = float(
            _get(data_cfg, "max_audio_seconds", DEFAULT_MAX_AUDIO_SECONDS)
        )
        self.min_prompt_seconds = float(_get(data_cfg, "min_prompt_seconds", 1.0))
        max_prompt_seconds = _get(
            data_cfg, "max_prompt_seconds", DEFAULT_MAX_PROMPT_SECONDS
        )
        self.max_prompt_seconds = (
            None if max_prompt_seconds in (None, "None") else float(max_prompt_seconds)
        )
        min_target_seconds = _get(data_cfg, "min_target_seconds", None)
        self.min_target_seconds = (
            None if min_target_seconds in (None, "None") else float(min_target_seconds)
        )
        max_target_seconds = _get(data_cfg, "max_target_seconds", None)
        self.max_target_seconds = (
            None if max_target_seconds in (None, "None") else float(max_target_seconds)
        )
        self.min_generated_frames = int(_get(data_cfg, "min_generated_frames", 8))
        if self.min_target_seconds is None:
            self.min_target_seconds = (
                self.min_generated_frames
                * self.mel_args["hop_size"]
                / self.mel_args["sampling_rate"]
            )
        if self.max_target_seconds is None:
            self.max_target_seconds = self.max_audio_seconds
        self.max_pair_seconds = float(
            _get(data_cfg, "max_pair_seconds", DEFAULT_MAX_PAIR_SECONDS)
        )
        self.min_pair_prompt_seconds = float(
            _get(data_cfg, "min_pair_prompt_seconds", 3.0)
        )
        self.sample_rate_16k = int(_get(data_cfg, "sample_rate_16k", 16000))
        self.sample_rate_mel = int(_get(data_cfg, "sample_rate_mel", self.mel_args["sampling_rate"]))
        self.feature_batch_size = max(1, int(_get(data_cfg, "feature_batch_size", 16)))
        self._resampler_cache: dict[
            tuple[int, int, str, torch.dtype], torchaudio.transforms.Resample
        ] = {}
        if self.sample_rate_mel != self.mel_args["sampling_rate"]:
            raise ValueError(
                "data.sample_rate_mel must match preprocess_params.sr "
                f"({self.sample_rate_mel} != {self.mel_args['sampling_rate']})"
            )
        s2mel_cfg = _get(cfg, "s2mel")
        dit_cfg = _get(s2mel_cfg, "DiT")
        model_mel_channels = int(_get(dit_cfg, "in_channels", self.mel_args["num_mels"]))
        if model_mel_channels != self.mel_args["num_mels"]:
            raise ValueError(
                "s2mel.DiT.in_channels must match preprocess mel bands "
                f"({model_mel_channels} != {self.mel_args['num_mels']})"
            )

        for module in (self.semantic_backend, self.semantic_decoder, self.campplus_model):
            if module is None:
                continue
            module.requires_grad_(False)

    def _module_device(self) -> torch.device:
        anchor = getattr(self, "_device_anchor", None)
        if isinstance(anchor, torch.Tensor):
            return anchor.device
        # Compatibility for tests and older code constructing the adapter
        # without running __init__.
        legacy_anchor = getattr(self, "semantic_mean", None)
        if isinstance(legacy_anchor, torch.Tensor):
            return legacy_anchor.device
        return torch.device("cpu")

    @torch.no_grad()
    def _semantic_from_waveforms(self, waveforms: list) -> list[torch.Tensor]:
        if self.semantic_backend is None:
            raise RuntimeError(
                "This feature adapter was initialized for precomputed semantic codes"
            )
        return self.semantic_backend.extract(waveforms)

    @torch.no_grad()
    def _semantic_from_codes(
        self,
        codes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        if self.semantic_decoder is None:
            raise RuntimeError("No MaskGCT semantic lookup table is loaded")
        if codes.ndim == 3 and codes.size(1) == 1:
            codes = codes[:, 0]
        if codes.ndim != 2 or lengths.ndim != 1 or lengths.size(0) != codes.size(0):
            raise ValueError("Semantic codes must be [B,T] with matching [B] lengths")
        device = self._module_device()
        decoded = self.semantic_decoder(codes.to(device))
        outputs = []
        for index, length in enumerate(lengths):
            value = int(length.item())
            if value <= 0 or value > decoded.size(1):
                raise ValueError(f"Invalid semantic code length {value}")
            outputs.append(decoded[index, :value].float())
        return outputs

    @torch.no_grad()
    def _semantic_from_audio(self, audio_16k: torch.Tensor) -> torch.Tensor:
        waveform = audio_16k.squeeze(0).detach().cpu().numpy()
        return self._semantic_from_waveforms([waveform])[0]

    @torch.no_grad()
    def _style_from_audio(self, audio_16k: torch.Tensor) -> torch.Tensor:
        device = self._module_device()
        if self.campplus_model is None:
            return torch.zeros(self.style_dim, device=device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=self.sample_rate_16k,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus_model(feat.unsqueeze(0)).squeeze(0).float()

    def _prepare_audio_batch(
        self,
        audio_paths: list[str],
        waveforms: list[torch.Tensor] | None,
        sample_rates: list[int] | None,
        *,
        max_audio_seconds: float | None | object = _USE_CONFIGURED_AUDIO_LIMIT,
    ) -> tuple[list[torch.Tensor], list[int]]:
        if (waveforms is None) != (sample_rates is None):
            raise ValueError("waveforms and sample_rates must be provided together")
        duration_limit = (
            self.max_audio_seconds
            if max_audio_seconds is _USE_CONFIGURED_AUDIO_LIMIT
            else max_audio_seconds
        )
        if waveforms is None or sample_rates is None:
            loaded = [_load_audio(path, duration_limit) for path in audio_paths]
            return [item[0] for item in loaded], [item[1] for item in loaded]
        if len(waveforms) != len(audio_paths) or len(sample_rates) != len(audio_paths):
            raise ValueError("Decoded audio batch must match audio_paths")

        prepared = []
        for audio, sample_rate in zip(waveforms, sample_rates, strict=True):
            if audio.ndim != 2:
                raise ValueError(f"Decoded waveform must be [channels, samples], got {audio.shape}")
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            if duration_limit is not None:
                audio = audio[:, : int(float(duration_limit) * sample_rate)]
            prepared.append(audio)
        return prepared, [int(sample_rate) for sample_rate in sample_rates]

    def _get_resampler(
        self,
        source_rate: int,
        target_rate: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torchaudio.transforms.Resample:
        key = (source_rate, target_rate, str(device), dtype)
        resampler = self._resampler_cache.get(key)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(
                source_rate,
                target_rate,
                dtype=dtype,
            ).to(device)
            resampler.eval()
            self._resampler_cache[key] = resampler
        return resampler

    @torch.no_grad()
    def _resample_waveform_batch(
        self,
        waveforms: list[torch.Tensor],
        sample_rates: list[int],
        target_rates: tuple[int, ...],
    ) -> dict[int, list[torch.Tensor]]:
        """Pad by source rate, transfer once, and batch-resample with cached kernels."""
        if not waveforms or len(waveforms) != len(sample_rates):
            raise ValueError("waveforms and sample_rates must have the same non-zero length")

        device = self._module_device()
        grouped_indices: dict[int, list[int]] = {}
        for index, sample_rate in enumerate(sample_rates):
            grouped_indices.setdefault(int(sample_rate), []).append(index)

        outputs: dict[int, list[torch.Tensor | None]] = {
            target_rate: [None] * len(waveforms) for target_rate in target_rates
        }
        for source_rate, indices in grouped_indices.items():
            source_waveforms = [waveforms[index].squeeze(0).float() for index in indices]
            source_lengths = [waveform.size(-1) for waveform in source_waveforms]
            padded = pad_sequence(source_waveforms, batch_first=True, padding_value=0.0).to(device)

            for target_rate in target_rates:
                if source_rate == target_rate:
                    resampled = padded
                else:
                    resampled = self._get_resampler(
                        source_rate,
                        target_rate,
                        device=device,
                        dtype=padded.dtype,
                    )(padded)
                for local_index, (global_index, source_length) in enumerate(
                    zip(indices, source_lengths, strict=True)
                ):
                    target_length = math.ceil(source_length * target_rate / source_rate)
                    outputs[target_rate][global_index] = resampled[
                        local_index : local_index + 1, :target_length
                    ]

        finalized: dict[int, list[torch.Tensor]] = {}
        for target_rate, items in outputs.items():
            if any(item is None for item in items):
                raise RuntimeError(f"Missing resampled waveform for {target_rate} Hz")
            finalized[target_rate] = [item for item in items if item is not None]
        return finalized

    @torch.no_grad()
    def extract_utterance_features(
        self,
        audio_paths: list[str],
        *,
        waveforms: list[torch.Tensor] | None = None,
        sample_rates: list[int] | None = None,
        semantic_codes: torch.Tensor | None = None,
        semantic_code_lens: torch.Tensor | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Extract per-utterance features. w2v-bert runs batched (chunked by
        ``feature_batch_size``); mel, campplus and codec quantization stay
        per-sample to keep parity with the single-utterance IndexTTS pipeline."""
        mels = []
        styles = []
        waveforms_16k = []
        if (semantic_codes is None) != (semantic_code_lens is None):
            raise ValueError("semantic_codes and semantic_code_lens must be provided together")
        code_mode = semantic_codes is not None
        use_style_condition = bool(getattr(self, "use_style_condition", False))
        need_16k = not code_mode or use_style_condition

        source_waveforms, source_rates = self._prepare_audio_batch(
            audio_paths, waveforms, sample_rates
        )
        resampled = self._resample_waveform_batch(
            source_waveforms,
            source_rates,
            (
                (self.sample_rate_mel, self.sample_rate_16k)
                if need_16k
                else (self.sample_rate_mel,)
            ),
        )
        for index, audio_mel in enumerate(resampled[self.sample_rate_mel]):
            mels.append(self.mel_spectrogram(audio_mel.float(), **self.mel_args).squeeze(0))
            if need_16k:
                audio_16k = resampled[self.sample_rate_16k][index]
                styles.append(self._style_from_audio(audio_16k))
                if not code_mode:
                    waveforms_16k.append(
                        audio_16k.squeeze(0).detach().cpu().numpy()
                    )
            else:
                styles.append(
                    torch.zeros(
                        int(getattr(self, "style_dim", 192)),
                        device=self._module_device(),
                    )
                )
        semantics: list[torch.Tensor] = []
        if semantic_codes is not None and semantic_code_lens is not None:
            semantics = self._semantic_from_codes(semantic_codes, semantic_code_lens)
        else:
            for start in range(0, len(waveforms_16k), self.feature_batch_size):
                semantics.extend(
                    self._semantic_from_waveforms(
                        waveforms_16k[start : start + self.feature_batch_size]
                    )
                )

        return [
            {"mel": mel, "semantic": semantic, "style": style}
            for mel, semantic, style in zip(mels, semantics, styles, strict=True)
        ]

    @torch.no_grad()
    def extract_random_split_from_audio_paths(
        self,
        audio_paths: list[str],
        *,
        waveforms: list[torch.Tensor] | None = None,
        sample_rates: list[int] | None = None,
        semantic_codes: torch.Tensor | None = None,
        semantic_code_lens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Randomly split each waveform into prompt and target before feature extraction."""
        device = self._module_device()
        prompt_mels = []
        target_mels = []
        styles = []
        segment_waveforms = []
        segment_sample_rates = []
        split_fractions: list[float] = []
        if (semantic_codes is None) != (semantic_code_lens is None):
            raise ValueError("semantic_codes and semantic_code_lens must be provided together")
        code_mode = semantic_codes is not None
        use_style_condition = bool(getattr(self, "use_style_condition", False))
        need_16k = not code_mode or use_style_condition

        target_floor_seconds = self.min_target_seconds
        if target_floor_seconds is None:
            target_floor_seconds = (
                self.min_generated_frames * self.mel_args["hop_size"] / self.sample_rate_mel
            )

        source_waveforms, source_rates = self._prepare_audio_batch(
            audio_paths, waveforms, sample_rates
        )
        for path, audio, sr in zip(
            audio_paths, source_waveforms, source_rates, strict=True
        ):
            min_prompt_samples = math.ceil(self.min_prompt_seconds * sr)
            min_target_samples = math.ceil(target_floor_seconds * sr)
            max_prompt_samples = (
                audio.size(-1)
                if self.max_prompt_seconds is None
                else int(self.max_prompt_seconds * sr)
            )
            lower = min_prompt_samples
            upper = min(max_prompt_samples, audio.size(-1) - min_target_samples)
            if upper < lower:
                duration = audio.size(-1) / sr
                raise ValueError(
                    f"Audio {path} is {duration:.3f}s, too short for a "
                    f"{self.min_prompt_seconds:g}s prompt and {target_floor_seconds:g}s target"
                )
            split_sample = lower if upper == lower else random.randint(lower, upper)
            split_fractions.append(split_sample / audio.size(-1))
            segment_waveforms.extend(
                [audio[:, :split_sample], audio[:, split_sample:]]
            )
            segment_sample_rates.extend([sr, sr])

        resampled = self._resample_waveform_batch(
            segment_waveforms,
            segment_sample_rates,
            (
                (self.sample_rate_mel, self.sample_rate_16k)
                if need_16k
                else (self.sample_rate_mel,)
            ),
        )
        segment_waveforms_mel = resampled[self.sample_rate_mel]
        segment_waveforms_16k = (
            resampled[self.sample_rate_16k] if need_16k else []
        )
        for index in range(len(audio_paths)):
            prompt_audio_mel = segment_waveforms_mel[2 * index]
            target_audio_mel = segment_waveforms_mel[2 * index + 1]
            prompt_mels.append(
                self.mel_spectrogram(prompt_audio_mel.float(), **self.mel_args).squeeze(0)
            )
            target_mels.append(
                self.mel_spectrogram(target_audio_mel.float(), **self.mel_args).squeeze(0)
            )

            if need_16k:
                styles.append(self._style_from_audio(segment_waveforms_16k[2 * index]))
            else:
                styles.append(
                    torch.zeros(int(getattr(self, "style_dim", 192)), device=device)
                )

        full_semantics = (
            self._semantic_from_codes(semantic_codes, semantic_code_lens)
            if semantic_codes is not None and semantic_code_lens is not None
            else None
        )
        segment_semantics: list[torch.Tensor] = []
        if full_semantics is None:
            semantic_waveforms_16k = [
                waveform.squeeze(0).detach().cpu().numpy()
                for waveform in segment_waveforms_16k
            ]
            for start in range(0, len(semantic_waveforms_16k), self.feature_batch_size):
                segment_semantics.extend(
                    self._semantic_from_waveforms(
                        semantic_waveforms_16k[start : start + self.feature_batch_size]
                    )
                )

        mels = []
        semantics = []
        prompt_lens = []
        prompt_semantic_lens = []
        for index, (prompt_mel, target_mel) in enumerate(
            zip(prompt_mels, target_mels, strict=True)
        ):
            if full_semantics is None:
                prompt_semantic = segment_semantics[2 * index]
                target_semantic = segment_semantics[2 * index + 1]
            else:
                full_semantic = full_semantics[index]
                if full_semantic.size(0) < 2:
                    raise ValueError("Random split requires at least two semantic code frames")
                prompt_semantic_len = round(
                    full_semantic.size(0) * split_fractions[index]
                )
                prompt_semantic_len = max(
                    1, min(full_semantic.size(0) - 1, prompt_semantic_len)
                )
                prompt_semantic = full_semantic[:prompt_semantic_len]
                target_semantic = full_semantic[prompt_semantic_len:]
            mels.append(torch.cat([prompt_mel, target_mel], dim=-1).transpose(0, 1))
            semantics.append(torch.cat([prompt_semantic, target_semantic], dim=0))
            prompt_lens.append(prompt_mel.size(-1))
            prompt_semantic_lens.append(prompt_semantic.size(0))

        mel_lens = torch.tensor([x.size(0) for x in mels], dtype=torch.long, device=device)
        semantic_lens = torch.tensor([x.size(0) for x in semantics], dtype=torch.long, device=device)
        mel = pad_sequence(mels, batch_first=True, padding_value=0.0).transpose(1, 2).to(device)
        semantic = pad_sequence(semantics, batch_first=True, padding_value=0.0).to(device)
        style = torch.stack(styles).to(device)
        prompt_lens_tensor = torch.tensor(prompt_lens, dtype=torch.long, device=device)
        return {
            "mel": mel,
            "mel_lens": mel_lens,
            "semantic": semantic,
            "semantic_lens": semantic_lens,
            "style": style,
            "prompt_lens": prompt_lens_tensor,
            "prompt_semantic_lens": torch.tensor(
                prompt_semantic_lens, dtype=torch.long, device=device
            ),
        }

    @torch.no_grad()
    def extract_from_audio_paths(
        self,
        audio_paths: list[str],
        *,
        waveforms: list[torch.Tensor] | None = None,
        sample_rates: list[int] | None = None,
        semantic_codes: torch.Tensor | None = None,
        semantic_code_lens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = self._module_device()
        features = self.extract_utterance_features(
            audio_paths,
            waveforms=waveforms,
            sample_rates=sample_rates,
            semantic_codes=semantic_codes,
            semantic_code_lens=semantic_code_lens,
        )

        mels = []
        semantics = []
        styles = []
        prompt_lens = []
        for item in features:
            mel = item["mel"]
            prompt_len = choose_prompt_len(
                mel.size(-1),
                hop_length=self.mel_args["hop_size"],
                sample_rate=self.sample_rate_mel,
                min_prompt_seconds=self.min_prompt_seconds,
                max_prompt_seconds=self.max_prompt_seconds,
                min_generated_frames=self.min_generated_frames,
                min_target_seconds=self.min_target_seconds,
            )
            mels.append(mel.transpose(0, 1))
            semantics.append(item["semantic"])
            styles.append(item["style"])
            prompt_lens.append(prompt_len)

        mel_lens = torch.tensor([x.size(0) for x in mels], dtype=torch.long, device=device)
        semantic_lens = torch.tensor([x.size(0) for x in semantics], dtype=torch.long, device=device)
        mel = pad_sequence(mels, batch_first=True, padding_value=0.0).transpose(1, 2).to(device)
        semantic = pad_sequence(semantics, batch_first=True, padding_value=0.0).to(device)
        style = torch.stack(styles).to(device)
        prompt_lens_tensor = torch.tensor(prompt_lens, dtype=torch.long, device=device)
        return {
            "mel": mel,
            "mel_lens": mel_lens,
            "semantic": semantic,
            "semantic_lens": semantic_lens,
            "style": style,
            "prompt_lens": prompt_lens_tensor,
        }

    @torch.no_grad()
    def extract_paired_from_audio_paths(
        self,
        prompt_audio_paths: list[str],
        target_audio_paths: list[str],
        *,
        prompt_waveforms: list[torch.Tensor] | None = None,
        prompt_sample_rates: list[int] | None = None,
        target_waveforms: list[torch.Tensor] | None = None,
        target_sample_rates: list[int] | None = None,
        singleton_splits: list[bool] | None = None,
        prompt_semantic_codes: torch.Tensor | None = None,
        prompt_semantic_code_lens: torch.Tensor | None = None,
        target_semantic_codes: torch.Tensor | None = None,
        target_semantic_code_lens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not prompt_audio_paths or len(prompt_audio_paths) != len(target_audio_paths):
            raise ValueError("Prompt and target path lists must have the same non-zero length")
        decoded = (prompt_waveforms, prompt_sample_rates, target_waveforms, target_sample_rates)
        if any(item is None for item in decoded) and not all(item is None for item in decoded):
            raise ValueError("All paired decoded waveform fields must be provided together")
        batch_size = len(prompt_audio_paths)
        if singleton_splits is None:
            singleton_splits = [False] * batch_size
        if len(singleton_splits) != batch_size:
            raise ValueError("singleton_splits must match the paired batch size")
        code_fields = (
            prompt_semantic_codes,
            prompt_semantic_code_lens,
            target_semantic_codes,
            target_semantic_code_lens,
        )
        if any(item is None for item in code_fields) and not all(
            item is None for item in code_fields
        ):
            raise ValueError("All paired semantic code fields must be provided together")
        code_mode = all(item is not None for item in code_fields)
        prompt_sources, prompt_rates = self._prepare_audio_batch(
            prompt_audio_paths,
            prompt_waveforms,
            prompt_sample_rates,
            max_audio_seconds=self.max_audio_seconds,
        )
        target_sources, target_rates = self._prepare_audio_batch(
            target_audio_paths,
            target_waveforms,
            target_sample_rates,
            max_audio_seconds=None,
        )

        min_target_seconds = self.min_target_seconds
        max_target_seconds = self.max_target_seconds
        if min_target_seconds is None or max_target_seconds is None:
            raise ValueError(
                "Paired extraction requires min_target_seconds and max_target_seconds"
            )
        if self.max_prompt_seconds is None:
            raise ValueError("Paired extraction requires max_prompt_seconds")

        prompt_segments: list[torch.Tensor] = []
        target_segments: list[torch.Tensor] = []
        segment_rates: list[int] = []
        singleton_code_splits: list[int | None] = []
        for index, singleton_split in enumerate(singleton_splits):
            if singleton_split:
                source = target_sources[index]
                sample_rate = target_rates[index]
                source_samples = source.size(-1)
                min_prompt_samples = math.ceil(self.min_pair_prompt_seconds * sample_rate)
                min_target_samples = math.ceil(min_target_seconds * sample_rate)
                max_prompt_samples = int(self.max_prompt_seconds * sample_rate)
                lower_sample = min_prompt_samples
                upper_sample = min(
                    max_prompt_samples,
                    source_samples - min_target_samples,
                )
                if upper_sample < lower_sample:
                    raise ValueError(
                        f"Singleton audio {target_audio_paths[index]} is too short for "
                        f"a {self.min_pair_prompt_seconds:g}s prompt and "
                        f"{min_target_seconds:g}s target"
                    )
                if code_mode:
                    assert target_semantic_code_lens is not None
                    code_length = int(target_semantic_code_lens[index])
                    lower_code = max(
                        1,
                        math.ceil(lower_sample * code_length / source_samples),
                    )
                    upper_code = min(
                        code_length - 1,
                        math.floor(upper_sample * code_length / source_samples),
                    )
                    if upper_code < lower_code:
                        raise ValueError(
                            f"Singleton audio {target_audio_paths[index]} has too few "
                            "semantic frames for the requested duration bounds"
                        )
                    split_code = (
                        lower_code
                        if upper_code == lower_code
                        else random.randint(lower_code, upper_code)
                    )
                    split_sample = round(split_code * source_samples / code_length)
                    split_sample = max(lower_sample, min(upper_sample, split_sample))
                    singleton_code_splits.append(split_code)
                else:
                    split_sample = (
                        lower_sample
                        if upper_sample == lower_sample
                        else random.randint(lower_sample, upper_sample)
                    )
                    singleton_code_splits.append(None)
                prompt_segment = source[:, :split_sample]
                target_segment = source[:, split_sample:]
            else:
                prompt_source = prompt_sources[index]
                prompt_rate = prompt_rates[index]
                target_segment = target_sources[index]
                target_rate = target_rates[index]
                prompt_samples = min(
                    prompt_source.size(-1),
                    int(self.max_prompt_seconds * prompt_rate),
                )
                if prompt_samples < math.ceil(
                    self.min_pair_prompt_seconds * prompt_rate
                ):
                    raise ValueError(
                        f"Prompt {prompt_audio_paths[index]} is shorter than "
                        f"{self.min_pair_prompt_seconds:g} seconds"
                    )
                target_seconds = target_segment.size(-1) / target_rate
                if target_seconds < min_target_seconds:
                    raise ValueError(
                        f"Target {target_audio_paths[index]} is shorter than "
                        f"{min_target_seconds:g} seconds"
                    )
                if target_seconds > max_target_seconds + (1.0 / target_rate):
                    raise ValueError(
                        f"Target {target_audio_paths[index]} is {target_seconds:.3f}s, "
                        f"exceeding the {max_target_seconds:g}s limit; targets are not cropped"
                    )
                prompt_segment = prompt_source[:, :prompt_samples]
                sample_rate = prompt_rate
                singleton_code_splits.append(None)

            prompt_segments.append(prompt_segment)
            target_segments.append(target_segment)
            segment_rates.extend(
                [
                    sample_rate,
                    target_rates[index],
                ]
            )

        interleaved_segments = [
            segment
            for pair in zip(prompt_segments, target_segments, strict=True)
            for segment in pair
        ]
        use_style_condition = bool(getattr(self, "use_style_condition", False))
        need_16k = not code_mode or use_style_condition
        resampled = self._resample_waveform_batch(
            interleaved_segments,
            segment_rates,
            (
                (self.sample_rate_mel, self.sample_rate_16k)
                if need_16k
                else (self.sample_rate_mel,)
            ),
        )
        mel_waveforms = resampled[self.sample_rate_mel]
        waveforms_16k = resampled[self.sample_rate_16k] if need_16k else []

        full_prompt_semantics: list[torch.Tensor] | None = None
        full_target_semantics: list[torch.Tensor] | None = None
        online_semantics: list[torch.Tensor] = []
        if code_mode:
            assert prompt_semantic_codes is not None
            assert prompt_semantic_code_lens is not None
            assert target_semantic_codes is not None
            assert target_semantic_code_lens is not None
            full_prompt_semantics = self._semantic_from_codes(
                prompt_semantic_codes,
                prompt_semantic_code_lens,
            )
            full_target_semantics = self._semantic_from_codes(
                target_semantic_codes,
                target_semantic_code_lens,
            )
        else:
            semantic_waveforms = [
                waveform.squeeze(0).detach().cpu().numpy()
                for waveform in waveforms_16k
            ]
            for start in range(0, len(semantic_waveforms), self.feature_batch_size):
                online_semantics.extend(
                    self._semantic_from_waveforms(
                        semantic_waveforms[start : start + self.feature_batch_size]
                    )
                )

        prompt_features: list[dict[str, torch.Tensor]] = []
        target_features: list[dict[str, torch.Tensor]] = []
        device = self._module_device()
        for index, singleton_split in enumerate(singleton_splits):
            prompt_mel = self.mel_spectrogram(
                mel_waveforms[2 * index].float(), **self.mel_args
            ).squeeze(0)
            target_mel = self.mel_spectrogram(
                mel_waveforms[2 * index + 1].float(), **self.mel_args
            ).squeeze(0)
            if code_mode:
                assert full_prompt_semantics is not None
                assert full_target_semantics is not None
                if singleton_split:
                    split_code = singleton_code_splits[index]
                    assert split_code is not None
                    full_semantic = full_target_semantics[index]
                    prompt_semantic = full_semantic[:split_code]
                    target_semantic = full_semantic[split_code:]
                else:
                    full_prompt = full_prompt_semantics[index]
                    prompt_keep = max(
                        1,
                        min(
                            full_prompt.size(0),
                            round(
                                full_prompt.size(0)
                                * prompt_segments[index].size(-1)
                                / prompt_sources[index].size(-1)
                            ),
                        ),
                    )
                    prompt_semantic = full_prompt[:prompt_keep]
                    target_semantic = full_target_semantics[index]
            else:
                prompt_semantic = online_semantics[2 * index]
                target_semantic = online_semantics[2 * index + 1]
            style = (
                self._style_from_audio(waveforms_16k[2 * index])
                if need_16k
                else torch.zeros(
                    int(getattr(self, "style_dim", 192)),
                    device=device,
                )
            )
            prompt_features.append(
                {"mel": prompt_mel, "semantic": prompt_semantic, "style": style}
            )
            target_features.append(
                {"mel": target_mel, "semantic": target_semantic, "style": style}
            )

        duration_budget = self.max_prompt_seconds + max_target_seconds
        return collate_paired_features(
            prompt_features,
            target_features,
            hop_length=self.mel_args["hop_size"],
            sample_rate=self.sample_rate_mel,
            max_pair_seconds=duration_budget,
            min_prompt_seconds=self.min_pair_prompt_seconds,
            min_generated_frames=self.min_generated_frames,
            is_precomputed=False,
        )


# Backward-compatible import for existing callers and tests.
IndexTTSFeatureAdapter = S2MelFeatureAdapter


def build_feature_adapter(
    cfg: Any,
    *,
    semantic_lookup_path: str | Path | None = None,
    semantic_lookup_sha256: str | None = None,
) -> S2MelFeatureAdapter:
    return S2MelFeatureAdapter(
        cfg,
        semantic_lookup_path=semantic_lookup_path,
        semantic_lookup_sha256=semantic_lookup_sha256,
    )


def move_feature_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in (
        "mel",
        "mel_lens",
        "semantic",
        "semantic_lens",
        "style",
        "prompt_lens",
        "prompt_semantic_lens",
    ):
        if key in out and isinstance(out[key], torch.Tensor):
            out[key] = out[key].to(device)
    return out
