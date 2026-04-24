from __future__ import annotations

from typing import Any

import numpy as np
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric


def threshold_predictions(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()


def compute_region_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    include_hd95: bool = True,
) -> dict[str, float]:
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    dice_metric(y_pred=predictions, y=targets)
    dice_values = dice_metric.aggregate().detach().cpu().numpy()
    dice_metric.reset()
    if include_hd95:
        hd_metric = HausdorffDistanceMetric(
            include_background=True,
            percentile=95,
            reduction="mean_batch",
        )
        hd_metric(y_pred=predictions, y=targets)
        hd_values = hd_metric.aggregate().detach().cpu().numpy()
        hd_metric.reset()
    else:
        hd_values = np.zeros_like(dice_values)
    return {
        "dice_tc": _safe_value(dice_values[0]),
        "dice_wt": _safe_value(dice_values[1]),
        "dice_et": _safe_value(dice_values[2]),
        "hd95_tc": _safe_value(hd_values[0]),
        "hd95_wt": _safe_value(hd_values[1]),
        "hd95_et": _safe_value(hd_values[2]),
        "mean_dice": _safe_value(np.mean(dice_values)),
        "mean_hd95": _safe_value(np.mean(hd_values)),
    }


def summarize_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = records[0].keys()
    return {key: float(np.nanmean([record[key] for record in records])) for key in keys}


def _safe_value(value: Any) -> float:
    scalar = float(value)
    if np.isnan(scalar) or np.isinf(scalar):
        return 0.0
    return scalar
