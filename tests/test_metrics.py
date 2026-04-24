from __future__ import annotations

import torch

from brain_mri_segmentation.ml.metrics import compute_region_metrics


def test_compute_region_metrics_can_skip_hd95() -> None:
    predictions = torch.ones((1, 3, 4, 4, 4), dtype=torch.float32)
    targets = torch.ones((1, 3, 4, 4, 4), dtype=torch.float32)

    metrics = compute_region_metrics(predictions, targets, include_hd95=False)

    assert metrics["dice_tc"] == 1.0
    assert metrics["dice_wt"] == 1.0
    assert metrics["dice_et"] == 1.0
    assert metrics["mean_dice"] == 1.0
    assert metrics["hd95_tc"] == 0.0
    assert metrics["hd95_wt"] == 0.0
    assert metrics["hd95_et"] == 0.0
    assert metrics["mean_hd95"] == 0.0
