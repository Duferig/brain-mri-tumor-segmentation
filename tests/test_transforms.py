from __future__ import annotations

import torch

from brain_mri_segmentation.ml.transforms import CustomConvertBratsLabelsd, _compact_cached_tensor


def test_compact_cached_tensor_breaks_large_backing_storage() -> None:
    full = torch.zeros((4, 240, 240, 155), dtype=torch.float32)
    cropped_view = full[:, 10:147, 20:186, 5:147]

    compact = _compact_cached_tensor(cropped_view)

    assert cropped_view.untyped_storage().nbytes() > cropped_view.numel() * cropped_view.element_size()
    assert compact.shape == cropped_view.shape
    assert compact.untyped_storage().nbytes() == compact.numel() * compact.element_size()


def test_custom_brats_labels_accepts_label_3_as_et() -> None:
    label = torch.tensor(
        [
            [[0, 1], [2, 3]],
            [[3, 0], [1, 2]],
        ],
        dtype=torch.uint8,
    )

    result = CustomConvertBratsLabelsd(keys="label")({"label": label})
    regions = result["label"]

    assert regions.shape == (3, 2, 2, 2)
    assert int(regions[0].sum().item()) == 4
    assert int(regions[1].sum().item()) == 6
    assert int(regions[2].sum().item()) == 2


def test_custom_brats_labels_accepts_legacy_label_4_as_et() -> None:
    label = torch.tensor(
        [
            [[0, 1], [2, 4]],
            [[4, 0], [1, 2]],
        ],
        dtype=torch.uint8,
    )

    result = CustomConvertBratsLabelsd(keys="label")({"label": label})
    regions = result["label"]

    assert regions.shape == (3, 2, 2, 2)
    assert int(regions[0].sum().item()) == 4
    assert int(regions[1].sum().item()) == 6
    assert int(regions[2].sum().item()) == 2
