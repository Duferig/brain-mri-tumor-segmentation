from __future__ import annotations

import pytest


np = pytest.importorskip("numpy")

from brain_mri_segmentation.ml.labels import (  # noqa: E402
    brats_label_to_regions,
    compute_voxel_statistics,
    regions_to_brats_label,
)


def test_brats_label_roundtrip() -> None:
    label = np.array(
        [
            [[0, 1], [2, 4]],
            [[4, 0], [1, 2]],
        ],
        dtype=np.uint8,
    )
    regions = brats_label_to_regions(label)
    restored = regions_to_brats_label(regions, threshold=0.5)
    assert restored.shape == label.shape
    assert np.array_equal(restored, label)


def test_voxel_statistics_match_brats_regions() -> None:
    label = np.array(
        [
            [[0, 1], [2, 4]],
            [[4, 0], [1, 2]],
        ],
        dtype=np.uint8,
    )
    stats = compute_voxel_statistics(label)
    assert stats["label_1"] == 2
    assert stats["label_2"] == 2
    assert stats["label_4"] == 2
    assert stats["TC"] == 4
    assert stats["WT"] == 6
    assert stats["ET"] == 2


def test_brats_2023_label_3_is_treated_as_enhancing_tumor() -> None:
    label = np.array(
        [
            [[0, 1], [2, 3]],
            [[3, 0], [1, 2]],
        ],
        dtype=np.uint8,
    )
    regions = brats_label_to_regions(label)
    restored = regions_to_brats_label(regions, threshold=0.5)
    stats = compute_voxel_statistics(label)
    assert regions.shape[0] == 3
    assert stats["label_3"] == 2
    assert stats["ET"] == 2
    assert np.sum(restored == 4) == 2
