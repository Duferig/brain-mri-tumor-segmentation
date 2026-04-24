from __future__ import annotations

from typing import Any

import torch

from monai.transforms import (
    Compose,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
)


def _compact_cached_tensor(data: torch.Tensor) -> torch.Tensor:
    # Break tensor views so PersistentDataset stores only the cropped region on disk.
    return data.clone(memory_format=torch.contiguous_format)


class CustomConvertBratsLabelsd(MapTransform):
    """Convert BraTS labels to TC/WT/ET regions while accepting label 3 or 4 as ET."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = dict(data)
        for key in self.keys:
            label = result[key]
            if not torch.is_tensor(label):
                label = torch.as_tensor(label)
            if label.ndim == 3:
                label = label.unsqueeze(0)

            et = torch.logical_or(label == 3, label == 4)
            tc = torch.logical_or(label == 1, et)
            wt = torch.logical_or(torch.logical_or(label == 1, label == 2), et)
            result[key] = torch.cat([tc, wt, et], dim=0).float()
        return result


def build_train_transform(roi_size: tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image"], data_type="tensor", track_meta=False),
            CustomConvertBratsLabelsd(keys="label"),
            EnsureTyped(
                keys=["label"],
                data_type="tensor",
                dtype=torch.float32,
                track_meta=False,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys=["image", "label"], func=_compact_cached_tensor),
            DeleteItemsd(
                keys=["case_id", "modality_map", "foreground_start_coord", "foreground_end_coord"],
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=2,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            EnsureTyped(
                keys=["image", "label"],
                data_type="tensor",
                dtype=torch.float32,
                track_meta=False,
            ),
        ]
    )


def build_eval_transform() -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image"], data_type="tensor", track_meta=False),
            CustomConvertBratsLabelsd(keys="label"),
            EnsureTyped(
                keys=["label"],
                data_type="tensor",
                dtype=torch.float32,
                track_meta=False,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Lambdad(keys=["image", "label"], func=_compact_cached_tensor),
            DeleteItemsd(
                keys=["case_id", "modality_map", "foreground_start_coord", "foreground_end_coord"],
            ),
            EnsureTyped(
                keys=["image", "label"],
                data_type="tensor",
                dtype=torch.float32,
                track_meta=False,
            ),
        ]
    )
