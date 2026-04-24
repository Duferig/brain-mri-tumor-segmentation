from __future__ import annotations

import torch

from brain_mri_segmentation.ml.transforms import _compact_cached_tensor


def test_compact_cached_tensor_breaks_large_backing_storage() -> None:
    full = torch.zeros((4, 240, 240, 155), dtype=torch.float32)
    cropped_view = full[:, 10:147, 20:186, 5:147]

    compact = _compact_cached_tensor(cropped_view)

    assert cropped_view.untyped_storage().nbytes() > cropped_view.numel() * cropped_view.element_size()
    assert compact.shape == cropped_view.shape
    assert compact.untyped_storage().nbytes() == compact.numel() * compact.element_size()
