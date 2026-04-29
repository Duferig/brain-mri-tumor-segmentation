from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from brain_mri_segmentation.api.schemas import (
    ExperimentSummary,
    ExperimentsResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PredictionResponse,
    PreviewImage,
)
from brain_mri_segmentation.api.service import create_request_upload_dir, persist_upload
from brain_mri_segmentation.ml.brats import DataValidationError
from brain_mri_segmentation.ml.config import InferenceAppConfig, load_inference_config
from brain_mri_segmentation.ml.labels import brats_label_to_regions
from brain_mri_segmentation.ml.inference import PredictionResult, SegmentationPredictor
from brain_mri_segmentation.ml.metrics import compute_region_metrics


CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


def get_default_config_path() -> str:
    return os.getenv("BRAIN_SEG_INFERENCE_CONFIG", "configs/inference.toml")


def create_app(config_path: str | None = None) -> FastAPI:
    runtime_config_path = config_path or get_default_config_path()
    config = load_inference_config(runtime_config_path)
    predictor = SegmentationPredictor(config)
    app = FastAPI(title="Brain MRI Segmentation API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(CORS_ORIGINS),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.config = config
    app.state.predictor = predictor
    app.mount("/artifacts", StaticFiles(directory=str(config.runtime.artifacts_dir)), name="artifacts")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs", status_code=307)

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", available_models=sorted(config.models.keys()))

    @app.get("/models", response_model=ModelsResponse)
    async def models() -> ModelsResponse:
        return ModelsResponse(
            models=[
                ModelInfo(
                    key=model.key,
                    name=model.name,
                    display_name=model.display_name,
                    roi_size=list(model.roi_size),
                    weights_available=model.weights.exists(),
                )
                for model in config.models.values()
            ]
        )

    @app.get("/experiments", response_model=ExperimentsResponse)
    async def experiments() -> ExperimentsResponse:
        return ExperimentsResponse(experiments=load_experiment_summaries(config))

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: Request,
        t1: UploadFile = File(...),
        t1ce: UploadFile = File(...),
        t2: UploadFile = File(...),
        flair: UploadFile = File(...),
        reference_seg: UploadFile | None = File(default=None),
        model: str = Form(config.ui.default_model),
    ) -> PredictionResponse:
        upload_dir = create_request_upload_dir(config.runtime.uploads_dir)
        saved_files = {
            "t1": await persist_upload(t1, upload_dir, "t1", config.runtime.max_upload_size_mb),
            "t1ce": await persist_upload(t1ce, upload_dir, "t1ce", config.runtime.max_upload_size_mb),
            "t2": await persist_upload(t2, upload_dir, "t2", config.runtime.max_upload_size_mb),
            "flair": await persist_upload(flair, upload_dir, "flair", config.runtime.max_upload_size_mb),
        }
        reference_path = None
        if reference_seg is not None:
            reference_path = await persist_upload(
                reference_seg,
                upload_dir,
                "reference_seg",
                config.runtime.max_upload_size_mb,
            )
        try:
            result = predictor.predict(saved_files, model_key=model)
            reference_metrics = (
                compare_prediction_to_reference(result.segmentation_path, reference_path)
                if reference_path is not None
                else None
            )
        except DataValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return build_response(result, config, request, reference_metrics=reference_metrics)

    return app


def build_response(
    result: PredictionResult,
    config: InferenceAppConfig,
    request: Request,
    reference_metrics: dict[str, float] | None = None,
) -> PredictionResponse:
    segmentation_rel = result.segmentation_path.relative_to(config.runtime.artifacts_dir)
    segmentation_url = str(request.base_url).rstrip("/") + f"/artifacts/{segmentation_rel.as_posix()}"
    previews = []
    for asset in result.preview_assets:
        original_rel = asset.original_path.relative_to(config.runtime.artifacts_dir)
        overlay_rel = asset.overlay_path.relative_to(config.runtime.artifacts_dir)
        previews.append(
            PreviewImage(
                plane=asset.plane,
                slice_index=asset.slice_index,
                modality=asset.modality,
                original_path=str(asset.original_path),
                original_url=str(request.base_url).rstrip("/")
                + f"/artifacts/{original_rel.as_posix()}",
                overlay_path=str(asset.overlay_path),
                overlay_url=str(request.base_url).rstrip("/")
                + f"/artifacts/{overlay_rel.as_posix()}",
                highlighted_labels=asset.highlighted_labels,
            )
        )
    return PredictionResponse(
        prediction_id=result.prediction_id,
        model_used=result.model_used,
        segmentation_path=str(result.segmentation_path),
        segmentation_url=segmentation_url,
        voxel_statistics=result.voxel_statistics,
        preview_images=previews,
        reference_metrics=reference_metrics,
    )


def load_experiment_summaries(config: InferenceAppConfig) -> list[ExperimentSummary]:
    metrics_dir = config.runtime.artifacts_dir / "metrics"
    if not metrics_dir.exists():
        return []
    configured_weights = {model.weights.resolve() for model in config.models.values()}
    summaries: list[ExperimentSummary] = []
    for path in sorted(metrics_dir.glob("*_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checkpoint_path = Path(str(payload.get("checkpoint_path", "")))
        checkpoint_candidates = []
        if checkpoint_path.name:
            checkpoint_candidates.append(checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_candidates.append(config.runtime.artifacts_dir.parent / checkpoint_path)
            checkpoint_candidates.append(config.runtime.artifacts_dir / "models" / checkpoint_path.name)
        is_configured_model = any(
            candidate.resolve() in configured_weights
            for candidate in checkpoint_candidates
        )
        if not is_configured_model:
            continue
        checkpoint_available = checkpoint_path.exists()
        if not checkpoint_available and checkpoint_path.name:
            checkpoint_available = (config.runtime.artifacts_dir / "models" / checkpoint_path.name).exists()
        summaries.append(
            ExperimentSummary(
                key=path.stem.removesuffix("_summary"),
                experiment_name=str(payload.get("experiment_name", path.stem)),
                model_name=str(payload.get("model_name", "")),
                best_epoch=_optional_int(payload.get("best_epoch")),
                mean_dice=_optional_float(payload.get("mean_dice")),
                dice_tc=_optional_float(payload.get("dice_tc")),
                dice_wt=_optional_float(payload.get("dice_wt")),
                dice_et=_optional_float(payload.get("dice_et")),
                checkpoint_available=checkpoint_available,
            )
        )
    summaries.sort(
        key=lambda item: item.mean_dice if item.mean_dice is not None else -1,
        reverse=True,
    )
    return summaries


def compare_prediction_to_reference(
    prediction_path: Path,
    reference_path: Path,
) -> dict[str, float]:
    prediction_label = _load_label_map(prediction_path)
    reference_label = _load_label_map(reference_path)
    if prediction_label.shape != reference_label.shape:
        raise ValueError(
            "Prediction and reference shapes do not match: "
            f"{prediction_label.shape} != {reference_label.shape}"
        )
    prediction_regions = torch.from_numpy(brats_label_to_regions(prediction_label)[None]).float()
    reference_regions = torch.from_numpy(brats_label_to_regions(reference_label)[None]).float()
    return compute_region_metrics(
        predictions=prediction_regions,
        targets=reference_regions,
        include_hd95=False,
    )


def _load_label_map(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    return np.asarray(image.get_fdata(dtype=np.float32)).round().astype(np.uint8)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the brain MRI segmentation API.")
    parser.add_argument("--config", default=get_default_config_path(), help="Path to inference TOML config.")
    return parser


def run() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_inference_config(args.config)
    uvicorn.run(
        create_app(args.config),
        host=config.runtime.api_host,
        port=config.runtime.api_port,
    )


app = create_app()


if __name__ == "__main__":
    run()
