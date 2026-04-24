from __future__ import annotations

from typing import Final

import numpy as np


REGION_ORDER: Final[tuple[str, str, str]] = ("TC", "WT", "ET")
MODALITY_ORDER: Final[tuple[str, str, str, str]] = ("t1", "t1ce", "t2", "flair")


def brats_label_to_regions(label: np.ndarray) -> np.ndarray:
    label = np.asarray(label)
    et_mask = np.logical_or(label == 3, label == 4)
    tc = np.logical_or(label == 1, et_mask)
    wt = np.logical_or(tc, label == 2)
    et = et_mask
    return np.stack((tc, wt, et), axis=0).astype(np.float32)


def regions_to_brats_label(regions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    regions = np.asarray(regions)
    if regions.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got shape {regions.shape}")
    tc = regions[0] > threshold
    wt = regions[1] > threshold
    et = regions[2] > threshold
    label = np.zeros(regions.shape[1:], dtype=np.uint8)
    label[np.logical_and(wt, np.logical_not(tc))] = 2
    label[np.logical_and(tc, np.logical_not(et))] = 1
    label[et] = 4
    return label


def compute_voxel_statistics(label: np.ndarray) -> dict[str, int]:
    label = np.asarray(label)
    et_count = int(np.sum(np.logical_or(label == 3, label == 4)))
    stats = {
        "label_1": int(np.sum(label == 1)),
        "label_2": int(np.sum(label == 2)),
        "label_3": int(np.sum(label == 3)),
        "label_4": int(np.sum(label == 4)),
    }
    stats["TC"] = stats["label_1"] + et_count
    stats["WT"] = stats["label_1"] + stats["label_2"] + et_count
    stats["ET"] = et_count
    return stats


def modality_index(modality: str) -> int:
    try:
        return MODALITY_ORDER.index(modality.lower())
    except ValueError as error:
        raise KeyError(f"Unknown modality '{modality}'") from error
