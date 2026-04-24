from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from brain_mri_segmentation.api.schemas import HealthResponse, PredictionResponse, PreviewImage
from brain_mri_segmentation.api.service import create_request_upload_dir, persist_upload
from brain_mri_segmentation.ml.brats import DataValidationError
from brain_mri_segmentation.ml.config import InferenceAppConfig, load_inference_config
from brain_mri_segmentation.ml.inference import PredictionResult, SegmentationPredictor


def get_default_config_path() -> str:
    return os.getenv("BRAIN_SEG_INFERENCE_CONFIG", "configs/inference.toml")


def create_app(config_path: str | None = None) -> FastAPI:
    runtime_config_path = config_path or get_default_config_path()
    config = load_inference_config(runtime_config_path)
    predictor = SegmentationPredictor(config)
    app = FastAPI(title="Brain MRI Segmentation API", version="0.1.0")
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

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(
        request: Request,
        t1: UploadFile = File(...),
        t1ce: UploadFile = File(...),
        t2: UploadFile = File(...),
        flair: UploadFile = File(...),
        model: str = Form(config.ui.default_model),
    ) -> PredictionResponse:
        upload_dir = create_request_upload_dir(config.runtime.uploads_dir)
        saved_files = {
            "t1": await persist_upload(t1, upload_dir, "t1", config.runtime.max_upload_size_mb),
            "t1ce": await persist_upload(t1ce, upload_dir, "t1ce", config.runtime.max_upload_size_mb),
            "t2": await persist_upload(t2, upload_dir, "t2", config.runtime.max_upload_size_mb),
            "flair": await persist_upload(flair, upload_dir, "flair", config.runtime.max_upload_size_mb),
        }
        try:
            result = predictor.predict(saved_files, model_key=model)
        except DataValidationError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except KeyError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return build_response(result, config, request)

    return app


def build_response(
    result: PredictionResult,
    config: InferenceAppConfig,
    request: Request,
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
    )


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
