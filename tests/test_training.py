from __future__ import annotations

import csv

import torch

from brain_mri_segmentation.ml.config import load_training_config
from brain_mri_segmentation.ml.training import _is_finite_tensor, build_loss, write_history


def test_write_history_accepts_records_with_different_fields(tmp_path) -> None:
    history_path = tmp_path / "history.csv"
    history = [
        {"epoch": 1.0, "train_loss": 0.8},
        {"epoch": 2.0, "train_loss": 0.7, "mean_dice": 0.5, "dice_tc": 0.4},
    ]

    write_history(history_path, history)

    with history_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    assert rows[0]["epoch"] == "1.0"
    assert rows[0]["train_loss"] == "0.8"
    assert rows[0]["mean_dice"] == ""
    assert rows[1]["mean_dice"] == "0.5"


def test_load_training_config_reads_optional_persistent_cache_dir(tmp_path) -> None:
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        """
[experiment]
name = "test"
output_dir = "artifacts/runs/test"

[data]
train_manifest = "artifacts/manifests/train.json"
val_manifest = "artifacts/manifests/val.json"
test_manifest = "artifacts/manifests/test.json"
num_workers = 0
val_num_workers = 1
persistent_workers = false
val_persistent_workers = false
prefetch_factor = 2
val_prefetch_factor = 1
cache_rate = 0.0
persistent_cache_dir = "artifacts/persistent_cache/test"
roi_size = [96, 96, 96]
batch_size = 1
sw_batch_size = 1
infer_overlap = 0.5

[training]
model_name = "baseline"
seed = 42
epochs = 1
patience = 1
learning_rate = 0.0002
weight_decay = 0.00001
amp = true
threshold = 0.5
val_interval = 1
pretrained_weights = "models/brats_mri_segmentation/models/model.pt"
pretrained_strict = false
loss_name = "dice_focal"
loss_weight = [1.0, 1.0, 3.0]
loss_gamma = 2.5
loss_lambda_dice = 1.0
loss_lambda_other = 1.5

[model]
in_channels = 4
out_channels = 3

[hardware]
device = "cuda"

[artifacts]
checkpoint_path = "artifacts/models/test.pt"
history_path = "artifacts/metrics/test.csv"
summary_path = "artifacts/metrics/test.json"
        """.strip(),
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config.data.persistent_cache_dir is not None
    assert config.data.persistent_cache_dir.name == "test"
    assert config.data.val_num_workers == 1
    assert config.data.val_persistent_workers is False
    assert config.data.val_prefetch_factor == 1
    assert config.training.compute_hd95 is True
    assert config.training.pretrained_weights is not None
    assert config.training.pretrained_weights.name == "model.pt"
    assert config.training.pretrained_strict is False
    assert config.training.loss_name == "dice_focal"
    assert config.training.loss_weight == (1.0, 1.0, 3.0)
    assert config.training.loss_gamma == 2.5
    assert config.training.loss_lambda_dice == 1.0
    assert config.training.loss_lambda_other == 1.5


def test_is_finite_tensor_detects_nan_values() -> None:
    assert _is_finite_tensor(torch.tensor(1.0))
    assert not _is_finite_tensor(torch.tensor(float("nan")))


def test_build_loss_supports_dice_focal(tmp_path) -> None:
    config_path = tmp_path / "train.toml"
    config_path.write_text(
        """
[experiment]
name = "test"
output_dir = "artifacts/runs/test"

[data]
train_manifest = "artifacts/manifests/train.json"
val_manifest = "artifacts/manifests/val.json"
test_manifest = "artifacts/manifests/test.json"
num_workers = 0
persistent_workers = false
prefetch_factor = 2
cache_rate = 0.0
persistent_cache_dir = "artifacts/persistent_cache/test"
roi_size = [96, 96, 96]
batch_size = 1
sw_batch_size = 1
infer_overlap = 0.5

[training]
model_name = "segresnet"
seed = 42
epochs = 1
patience = 1
learning_rate = 0.00001
weight_decay = 0.00001
amp = false
threshold = 0.5
val_interval = 1
loss_name = "dice_focal"
loss_weight = [1.0, 1.0, 3.0]
loss_gamma = 2.0
loss_lambda_dice = 1.0
loss_lambda_other = 1.5

[model]
in_channels = 4
out_channels = 3

[hardware]
device = "cpu"

[artifacts]
checkpoint_path = "artifacts/models/test.pt"
history_path = "artifacts/metrics/test.csv"
summary_path = "artifacts/metrics/test.json"
        """.strip(),
        encoding="utf-8",
    )

    config = load_training_config(config_path)
    loss_fn = build_loss(config, torch.device("cpu"))

    assert loss_fn.__class__.__name__ == "DiceFocalLoss"
