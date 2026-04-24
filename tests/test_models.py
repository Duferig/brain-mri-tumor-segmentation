from __future__ import annotations

import torch

from brain_mri_segmentation.ml.models import build_model, load_weights


def test_build_improved_model_is_compatible_with_installed_monai() -> None:
    model = build_model(
        model_name="improved",
        in_channels=4,
        out_channels=3,
        roi_size=(64, 64, 64),
    )

    assert model is not None


def test_build_segresnet_model_is_supported() -> None:
    model = build_model(
        model_name="segresnet",
        in_channels=4,
        out_channels=3,
        roi_size=(96, 96, 96),
    )

    assert model is not None


def test_load_weights_accepts_state_dict_checkpoints(tmp_path) -> None:
    source_model = build_model(
        model_name="segresnet",
        in_channels=4,
        out_channels=3,
        roi_size=(96, 96, 96),
    )
    checkpoint_path = tmp_path / "bundle_checkpoint.pt"
    torch.save({"state_dict": source_model.state_dict()}, checkpoint_path)

    restored_model = build_model(
        model_name="segresnet",
        in_channels=4,
        out_channels=3,
        roi_size=(96, 96, 96),
    )
    checkpoint = load_weights(restored_model, checkpoint_path, map_location="cpu")

    assert "state_dict" in checkpoint
    for key, value in source_model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[key])
