from __future__ import annotations

import torch

from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd,
    DeleteItemsd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandSpatialCropd,
)


def _compact_cached_tensor(data: torch.Tensor) -> torch.Tensor:
    # Break tensor views so PersistentDataset stores only the cropped region on disk.
    return data.clone(memory_format=torch.contiguous_format)


def build_train_transform(roi_size: tuple[int, int, int]) -> Compose:
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image"], data_type="tensor", track_meta=False),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
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
            RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
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
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
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
