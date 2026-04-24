from __future__ import annotations

from collections.abc import Mapping
import inspect
from pathlib import Path
from typing import Any

import torch
from monai.networks.nets import SegResNet, SwinUNETR, UNet


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_model(
    model_name: str,
    in_channels: int,
    out_channels: int,
    roi_size: tuple[int, int, int],
) -> torch.nn.Module:
    if model_name == "baseline":
        return UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
    if model_name == "improved":
        swinunetr_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "feature_size": 24,
            "use_checkpoint": True,
        }
        if "img_size" in inspect.signature(SwinUNETR).parameters:
            swinunetr_kwargs["img_size"] = roi_size
        return SwinUNETR(**swinunetr_kwargs)
    if model_name == "segresnet":
        return SegResNet(
            spatial_dims=3,
            blocks_down=(1, 2, 2, 4),
            blocks_up=(1, 1, 1),
            init_filters=16,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=0.2,
        )
    raise ValueError(f"Unknown model name '{model_name}'")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "metadata": metadata or {},
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def _extract_state_dict(checkpoint: dict[str, Any]) -> Mapping[str, Any]:
    if "model_state" in checkpoint and isinstance(checkpoint["model_state"], Mapping):
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a valid state dict.")
    if state_dict and all(isinstance(key, str) and key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_weights(
    model: torch.nn.Module,
    path: str | Path,
    map_location: torch.device | str,
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint
