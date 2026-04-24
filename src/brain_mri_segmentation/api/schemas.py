from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    available_models: list[str]


class PreviewImage(BaseModel):
    plane: str
    slice_index: int
    modality: str
    original_path: str
    original_url: str
    overlay_path: str
    overlay_url: str
    highlighted_labels: list[str] = Field(default_factory=list)


class ModelInfo(BaseModel):
    key: str
    name: str
    display_name: str
    roi_size: list[int]
    weights_available: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class ExperimentSummary(BaseModel):
    key: str
    experiment_name: str
    model_name: str
    best_epoch: int | None = None
    mean_dice: float | None = None
    dice_tc: float | None = None
    dice_wt: float | None = None
    dice_et: float | None = None
    checkpoint_available: bool = False


class ExperimentsResponse(BaseModel):
    experiments: list[ExperimentSummary]


class PredictionResponse(BaseModel):
    prediction_id: str
    model_used: str
    segmentation_path: str
    segmentation_url: str
    voxel_statistics: dict[str, int] = Field(default_factory=dict)
    preview_images: list[PreviewImage] = Field(default_factory=list)
    reference_metrics: dict[str, float] | None = None
