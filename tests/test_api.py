from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("fastapi")
pytest.importorskip("numpy")
pytest.importorskip("nibabel")
pytest.importorskip("torch")
pytest.importorskip("monai")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from brain_mri_segmentation.api.main import create_app  # noqa: E402


def _write_inference_config(tmp_path: Path) -> Path:
    artifacts = tmp_path / "artifacts"
    for relative in ("predictions", "uploads", "models"):
        (artifacts / relative).mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "inference.toml"
    config_path.write_text(
        "\n".join(
            [
                "[runtime]",
                f'artifacts_dir = "{artifacts.as_posix()}"',
                f'predictions_dir = "{(artifacts / "predictions").as_posix()}"',
                f'uploads_dir = "{(artifacts / "uploads").as_posix()}"',
                'api_host = "127.0.0.1"',
                "api_port = 8000",
                'preview_modality = "flair"',
                "preview_opacity = 0.45",
                "max_upload_size_mb = 10",
                "",
                "[models.improved]",
                'name = "improved"',
                'display_name = "Improved SwinUNETR"',
                f'weights = "{(artifacts / "models" / "improved.pt").as_posix()}"',
                "roi_size = [96, 96, 96]",
                'device = "cpu"',
                "",
                "[ui]",
                'api_url = "http://127.0.0.1:8000"',
                'default_model = "improved"',
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _write_label(path: Path) -> None:
    label = np.zeros((6, 6, 6), dtype=np.uint8)
    label[1:4, 1:4, 1:4] = 1
    label[2:5, 2:5, 2:5] = 2
    label[3:5, 3:5, 3:5] = 4
    nib.save(nib.Nifti1Image(label, affine=np.eye(4)), str(path))


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_api_health_and_predict_contract(tmp_path, monkeypatch) -> None:
    config_path = _write_inference_config(tmp_path)
    metrics_dir = tmp_path / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "improved_summary.json").write_text(
        (
            '{"experiment_name": "improved-swinunetr", "model_name": "improved", '
            '"best_epoch": 23, "mean_dice": 0.8747, "dice_tc": null, '
            '"dice_wt": null, "dice_et": null, '
            f'"checkpoint_path": "{(tmp_path / "artifacts" / "models" / "improved.pt").as_posix()}"'
            "}"
        ),
        encoding="utf-8",
    )
    app = create_app(str(config_path))

    def fake_predict(saved_files, model_key="improved"):
        prediction_dir = tmp_path / "artifacts" / "predictions" / "case123"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        segmentation_path = prediction_dir / "seg.nii.gz"
        _write_label(segmentation_path)
        original_path = prediction_dir / "axial_10_original.png"
        overlay_path = prediction_dir / "axial_10_overlay.png"
        original_path.write_bytes(b"original")
        overlay_path.write_bytes(b"overlay")
        return SimpleNamespace(
            prediction_id="case123",
            model_used=model_key,
            segmentation_path=segmentation_path,
            preview_assets=[
                SimpleNamespace(
                    plane="axial",
                    slice_index=10,
                    modality="FLAIR",
                    original_path=original_path,
                    overlay_path=overlay_path,
                    highlighted_labels=["Label 1 / TC: non-enhancing tumor core"],
                )
            ],
            voxel_statistics={"WT": 12, "TC": 8, "ET": 4},
        )

    monkeypatch.setattr(app.state.predictor, "predict", fake_predict)
    client = TestClient(app)

    root_response = client.get("/", follow_redirects=False)
    assert root_response.status_code == 307
    assert root_response.headers["location"] == "/docs"

    favicon_response = client.get("/favicon.ico")
    assert favicon_response.status_code == 204

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json()["available_models"] == ["improved"]
    assert health_response.headers.get("access-control-allow-origin") is None

    cors_response = client.get("/health", headers={"Origin": "http://127.0.0.1:5173"})
    assert cors_response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"

    models_response = client.get("/models")
    assert models_response.status_code == 200
    models_payload = models_response.json()
    assert [item["key"] for item in models_payload["models"]] == ["improved"]
    assert models_payload["models"][0]["display_name"] == "Improved SwinUNETR"
    assert models_payload["models"][0]["roi_size"] == [96, 96, 96]
    assert models_payload["models"][0]["weights_available"] is False

    experiments_response = client.get("/experiments")
    assert experiments_response.status_code == 200
    experiments = experiments_response.json()["experiments"]
    assert experiments[0]["experiment_name"] == "improved-swinunetr"
    assert experiments[0]["mean_dice"] == 0.8747
    assert experiments[0]["checkpoint_available"] is False

    files = {
        "t1": ("t1.nii.gz", b"fake", "application/octet-stream"),
        "t1ce": ("t1ce.nii.gz", b"fake", "application/octet-stream"),
        "t2": ("t2.nii.gz", b"fake", "application/octet-stream"),
        "flair": ("flair.nii.gz", b"fake", "application/octet-stream"),
    }
    predict_response = client.post("/predict", files=files)
    assert predict_response.status_code == 200
    payload = predict_response.json()
    assert payload["prediction_id"] == "case123"
    assert payload["model_used"] == "improved"
    assert payload["segmentation_url"].endswith("/artifacts/predictions/case123/seg.nii.gz")
    assert payload["preview_images"][0]["modality"] == "FLAIR"
    assert payload["preview_images"][0]["original_url"].endswith(
        "/artifacts/predictions/case123/axial_10_original.png"
    )
    assert payload["preview_images"][0]["overlay_url"].endswith(
        "/artifacts/predictions/case123/axial_10_overlay.png"
    )
    assert payload["preview_images"][0]["highlighted_labels"] == [
        "Label 1 / TC: non-enhancing tumor core"
    ]
    assert payload["reference_metrics"] is None

    reference_path = tmp_path / "reference.nii.gz"
    _write_label(reference_path)
    reference_files = {
        **files,
        "reference_seg": ("seg.nii.gz", _read_bytes(reference_path), "application/octet-stream"),
    }
    reference_response = client.post("/predict", files=reference_files)
    assert reference_response.status_code == 200
    reference_payload = reference_response.json()
    assert reference_payload["reference_metrics"]["mean_dice"] == 1.0
    assert reference_payload["reference_metrics"]["dice_wt"] == 1.0
