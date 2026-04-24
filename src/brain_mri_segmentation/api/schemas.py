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


class PredictionResponse(BaseModel):
    prediction_id: str
    model_used: str
    segmentation_path: str
    segmentation_url: str
    voxel_statistics: dict[str, int] = Field(default_factory=dict)
    preview_images: list[PreviewImage] = Field(default_factory=list)
