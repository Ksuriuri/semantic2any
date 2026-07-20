from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


def _unwrap_state_dict(state_dict: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def model_net_state(model) -> dict[str, dict[str, torch.Tensor]]:
    """Return an IndexTTS-compatible ``net`` dictionary."""
    return {name: module.state_dict() for name, module in model.models.items()}


def save_compatible_checkpoint(
    path: str | Path,
    model,
    *,
    epoch: int = 0,
    step: int = 0,
    config: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "net": model_net_state(model),
        "epoch": epoch,
        "iters": step,
    }
    if config is not None:
        payload["config"] = config
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_compatible_checkpoint(
    model,
    path: str | Path,
    *,
    strict: bool = False,
    ignore_modules: tuple[str, ...] = (),
) -> tuple[int, int]:
    state = torch.load(path, map_location="cpu")
    params = state.get("net", state)
    for name, module in model.models.items():
        if name not in params or name in ignore_modules:
            continue
        module_state = _unwrap_state_dict(params[name])
        if name == "length_regulator" and "content_in_proj.weight" in module_state:
            checkpoint_dim = int(module_state["content_in_proj.weight"].shape[1])
            current_weight = module.state_dict().get("content_in_proj.weight")
            current_dim = int(current_weight.shape[1]) if current_weight is not None else None
            if current_dim != checkpoint_dim:
                raise ValueError(
                    "Semantic codec checkpoint is incompatible: "
                    f"checkpoint input dim={checkpoint_dim}, config input dim={current_dim}. "
                    "MaskGCT uses 1024 dims and SAC raw semantic embeddings use 1280 dims."
                )
        module.load_state_dict(module_state, strict=strict)
    return int(state.get("epoch", 0)), int(state.get("iters", state.get("step", 0)))
