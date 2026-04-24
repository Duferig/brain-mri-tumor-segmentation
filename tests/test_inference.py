from __future__ import annotations

from pathlib import Path

import torch

from brain_mri_segmentation.ml.config import load_inference_config
from brain_mri_segmentation.ml.inference import SegmentationPredictor
from brain_mri_segmentation.ml.models import build_model


def _write_model_checkpoint(path: Path, model_name: str) -> None:
    model = build_model(
        model_name=model_name,
        in_channels=4,
        out_channels=3,
        roi_size=(96, 96, 96),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)


def _write_inference_config(tmp_path: Path) -> Path:
    artifacts = tmp_path / "artifacts"
    predictions = artifacts / "predictions"
    uploads = artifacts / "uploads"
    models = artifacts / "models"
    predictions.mkdir(parents=True, exist_ok=True)
    uploads.mkdir(parents=True, exist_ok=True)
    _write_model_checkpoint(models / "baseline.pt", "baseline")
    _write_model_checkpoint(models / "transfer.pt", "segresnet")
    _write_model_checkpoint(models / "transfer_v2.pt", "segresnet")

    config_path = tmp_path / "inference.toml"
    config_path.write_text(
        "\n".join(
            [
                "[runtime]",
                f'artifacts_dir = "{artifacts.as_posix()}"',
                f'predictions_dir = "{predictions.as_posix()}"',
                f'uploads_dir = "{uploads.as_posix()}"',
                'api_host = "127.0.0.1"',
                "api_port = 8000",
                'preview_modality = "flair"',
                "preview_opacity = 0.45",
                "max_upload_size_mb = 10",
                "",
                "[models.baseline]",
                'name = "baseline"',
                'display_name = "Baseline 3D U-Net"',
                f'weights = "{(models / "baseline.pt").as_posix()}"',
                "roi_size = [96, 96, 96]",
                'device = "cpu"',
                "",
                "[models.transfer]",
                'name = "segresnet"',
                'display_name = "Transfer SegResNet"',
                f'weights = "{(models / "transfer.pt").as_posix()}"',
                "roi_size = [96, 96, 96]",
                'device = "cpu"',
                "",
                "[models.transfer_v2]",
                'name = "segresnet"',
                'display_name = "Transfer SegResNet v2 (ET Refine)"',
                f'weights = "{(models / "transfer_v2.pt").as_posix()}"',
                "roi_size = [96, 96, 96]",
                'device = "cpu"',
                "",
                "[ui]",
                'api_url = "http://127.0.0.1:8000"',
                'default_model = "baseline"',
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_predictor_loads_architecture_from_model_config(tmp_path) -> None:
    predictor = SegmentationPredictor(load_inference_config(_write_inference_config(tmp_path)))

    baseline_model, baseline_device, _ = predictor._get_model_bundle("baseline")
    transfer_model, transfer_device, _ = predictor._get_model_bundle("transfer")
    transfer_v2_model, transfer_v2_device, _ = predictor._get_model_bundle("transfer_v2")

    assert baseline_model.__class__.__name__ == "UNet"
    assert transfer_model.__class__.__name__ == "SegResNet"
    assert transfer_v2_model.__class__.__name__ == "SegResNet"
    assert baseline_device.type == "cpu"
    assert transfer_device.type == "cpu"
    assert transfer_v2_device.type == "cpu"
    assert predictor.config.models["baseline"].display_name == "Baseline 3D U-Net"
    assert predictor.config.models["transfer"].display_name == "Transfer SegResNet"
    assert predictor.config.models["transfer_v2"].display_name == "Transfer SegResNet v2 (ET Refine)"


def test_predictor_caches_loaded_models(tmp_path) -> None:
    predictor = SegmentationPredictor(load_inference_config(_write_inference_config(tmp_path)))

    first_model, _, _ = predictor._get_model_bundle("baseline")
    second_model, _, _ = predictor._get_model_bundle("baseline")

    assert first_model is second_model
