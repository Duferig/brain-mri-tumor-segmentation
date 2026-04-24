from __future__ import annotations

from brain_mri_segmentation.ml.brats import load_manifest
from brain_mri_segmentation.ml.config import load_training_config
from brain_mri_segmentation.ml.training import build_dataset
from brain_mri_segmentation.ml.transforms import build_train_transform


def test_train_dataset_drops_non_batch_keys() -> None:
    config = load_training_config("configs/train_baseline_3060_12gb_windows_workers.toml")
    records = load_manifest(config.data.train_manifest)
    dataset = build_dataset(
        records[:1],
        build_train_transform(config.data.roi_size),
        cache_rate=0.0,
        persistent_cache_dir=None,
    )

    item = dataset[0]

    assert "image" in item
    assert "label" in item
    assert "case_id" not in item
    assert "modality_map" not in item
    assert "foreground_start_coord" not in item
    assert "foreground_end_coord" not in item
