from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

from brain_mri_segmentation.ml.brats import CaseFiles, DataValidationError, validate_case_files
from brain_mri_segmentation.ml.config import InferenceAppConfig, InferenceModelConfig
from brain_mri_segmentation.ml.labels import compute_voxel_statistics, modality_index, regions_to_brats_label
from brain_mri_segmentation.ml.models import build_model, load_weights, resolve_device


LABEL_COLORS = {
    1: np.array([255, 191, 0], dtype=np.uint8),
    2: np.array([0, 153, 255], dtype=np.uint8),
    4: np.array([255, 64, 64], dtype=np.uint8),
}

LABEL_DESCRIPTIONS = {
    1: "Label 1 / TC: non-enhancing tumor core",
    2: "Label 2 / WT-only: peritumoral edema",
    4: "Label 4 / ET: enhancing tumor",
}


@dataclass(slots=True)
class PreviewAsset:
    plane: str
    slice_index: int
    modality: str
    original_path: Path
    overlay_path: Path
    highlighted_labels: list[str]


@dataclass(slots=True)
class PredictionResult:
    prediction_id: str
    model_used: str
    segmentation_path: Path
    preview_assets: list[PreviewAsset]
    voxel_statistics: dict[str, int]


class SegmentationPredictor:
    def __init__(self, config: InferenceAppConfig) -> None:
        self.config = config
        self.config.runtime.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.config.runtime.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, tuple[torch.nn.Module, torch.device, InferenceModelConfig]] = {}

    def predict(self, modalities: dict[str, Path], model_key: str = "baseline") -> PredictionResult:
        if model_key not in self.config.models:
            raise KeyError(f"Unknown model '{model_key}'")
        case = CaseFiles(
            case_id="uploaded-case",
            t1=modalities["t1"],
            t1ce=modalities["t1ce"],
            t2=modalities["t2"],
            flair=modalities["flair"],
            label=None,
        )
        validate_case_files(case, require_label=False)
        image_stack, reference_image = load_modalities(case)
        cropped_stack, bbox = crop_foreground(image_stack)
        normalized = normalize_modalities(cropped_stack)
        model, device, model_config = self._get_model_bundle(model_key)
        tensor = torch.from_numpy(normalized[None]).to(device)
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=tensor,
                roi_size=model_config.roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )
        prediction = (torch.sigmoid(logits) >= 0.5).float()[0].detach().cpu().numpy()
        cropped_label = regions_to_brats_label(prediction)
        label_map = restore_crop(cropped_label, bbox, image_stack.shape[1:])
        prediction_id = uuid.uuid4().hex[:12]
        output_dir = self.config.runtime.predictions_dir / prediction_id
        output_dir.mkdir(parents=True, exist_ok=True)
        segmentation_path = output_dir / "seg.nii.gz"
        save_label_map(segmentation_path, label_map, reference_image)
        previews = create_previews(
            image_stack=image_stack,
            label_map=label_map,
            output_dir=output_dir,
            modality=self.config.runtime.preview_modality,
            opacity=self.config.runtime.preview_opacity,
        )
        return PredictionResult(
            prediction_id=prediction_id,
            model_used=model_key,
            segmentation_path=segmentation_path,
            preview_assets=previews,
            voxel_statistics=compute_voxel_statistics(label_map),
        )

    def _get_model_bundle(
        self,
        model_key: str,
    ) -> tuple[torch.nn.Module, torch.device, InferenceModelConfig]:
        cached = self._models.get(model_key)
        if cached is not None:
            return cached
        model_config = self.config.models[model_key]
        if not model_config.weights.exists():
            raise FileNotFoundError(
                f"Weights for model '{model_key}' not found at {model_config.weights}. "
                "Train the model or update configs/inference.toml."
            )
        device = resolve_device(model_config.device)
        model = build_model(
            model_name=model_config.name,
            in_channels=4,
            out_channels=3,
            roi_size=model_config.roi_size,
        )
        load_weights(model, model_config.weights, map_location=device)
        model.to(device)
        model.eval()
        bundle = (model, device, model_config)
        self._models[model_key] = bundle
        return bundle


def load_modalities(case: CaseFiles) -> tuple[np.ndarray, nib.Nifti1Image]:
    volumes = []
    reference_image: nib.Nifti1Image | None = None
    for modality_path in (case.t1, case.t1ce, case.t2, case.flair):
        image = nib.load(str(modality_path))
        data = image.get_fdata(dtype=np.float32)
        volumes.append(data)
        if reference_image is None:
            reference_image = image
    if reference_image is None:
        raise DataValidationError("No MRI volumes were loaded")
    return np.stack(volumes, axis=0), reference_image


def normalize_modalities(image_stack: np.ndarray) -> np.ndarray:
    normalized = image_stack.astype(np.float32, copy=True)
    for channel in range(normalized.shape[0]):
        data = normalized[channel]
        mask = data != 0
        if not np.any(mask):
            continue
        mean = float(data[mask].mean())
        std = float(data[mask].std())
        std = std if std > 1e-6 else 1.0
        normalized[channel, mask] = (data[mask] - mean) / std
    return normalized


def crop_foreground(image_stack: np.ndarray) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    foreground = np.any(image_stack != 0, axis=0)
    if not np.any(foreground):
        depth, height, width = image_stack.shape[1:]
        bbox = (slice(0, depth), slice(0, height), slice(0, width))
        return image_stack, bbox
    coords = np.argwhere(foreground)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    bbox = tuple(slice(int(mins[index]), int(maxs[index])) for index in range(3))
    return image_stack[(slice(None), *bbox)], bbox


def restore_crop(
    cropped_label: np.ndarray,
    bbox: tuple[slice, slice, slice],
    original_shape: tuple[int, int, int],
) -> np.ndarray:
    restored = np.zeros(original_shape, dtype=np.uint8)
    restored[bbox] = cropped_label
    return restored


def save_label_map(path: Path, label_map: np.ndarray, reference_image: nib.Nifti1Image) -> None:
    output = nib.Nifti1Image(
        label_map.astype(np.uint8),
        affine=reference_image.affine,
        header=reference_image.header,
    )
    nib.save(output, str(path))


def create_previews(
    image_stack: np.ndarray,
    label_map: np.ndarray,
    output_dir: Path,
    modality: str,
    opacity: float,
) -> list[PreviewAsset]:
    channel = image_stack[modality_index(modality)]
    slices = choose_preview_slices(label_map)
    assets: list[PreviewAsset] = []
    for plane, slice_index in slices.items():
        base_slice, label_slice = slice_volume(channel, label_map, plane, slice_index)
        original = render_original(base_slice)
        overlay = render_overlay(original, label_slice, opacity=opacity)
        original_path = output_dir / f"{plane}_{slice_index}_original.png"
        overlay_path = output_dir / f"{plane}_{slice_index}_overlay.png"
        Image.fromarray(original).save(original_path)
        Image.fromarray(overlay).save(overlay_path)
        assets.append(
            PreviewAsset(
                plane=plane,
                slice_index=slice_index,
                modality=modality.upper(),
                original_path=original_path,
                overlay_path=overlay_path,
                highlighted_labels=describe_slice_labels(label_slice),
            )
        )
    return assets


def choose_preview_slices(label_map: np.ndarray) -> dict[str, int]:
    foreground = np.argwhere(label_map > 0)
    if foreground.size == 0:
        center = np.array(label_map.shape) // 2
    else:
        center = np.round(foreground.mean(axis=0)).astype(int)
    return {"axial": int(center[2]), "coronal": int(center[1]), "sagittal": int(center[0])}


def render_original(base_slice: np.ndarray) -> np.ndarray:
    normalized = normalize_slice(base_slice)
    return np.stack([normalized, normalized, normalized], axis=-1)


def render_overlay(
    original_rgb: np.ndarray,
    label_slice: np.ndarray,
    opacity: float,
) -> np.ndarray:
    rgb = original_rgb.copy()
    overlay = np.zeros_like(rgb)
    for label_value, color in LABEL_COLORS.items():
        overlay[label_slice == label_value] = color
    blended = np.where(
        overlay.sum(axis=-1, keepdims=True) > 0,
        (rgb.astype(np.float32) * (1.0 - opacity) + overlay.astype(np.float32) * opacity),
        rgb.astype(np.float32),
    )
    contour_mask = np.zeros(label_slice.shape, dtype=bool)
    contour_rgb = np.zeros_like(rgb)
    for label_value, color in LABEL_COLORS.items():
        label_mask = label_slice == label_value
        if not np.any(label_mask):
            continue
        contour = label_mask & ~binary_erosion(label_mask, structure=np.ones((3, 3), dtype=bool))
        contour = binary_dilation(contour, structure=np.ones((3, 3), dtype=bool))
        contour_mask |= contour
        contour_rgb[contour] = color
    blended[contour_mask] = contour_rgb[contour_mask].astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)


def describe_slice_labels(label_slice: np.ndarray) -> list[str]:
    labels = [int(label) for label in np.unique(label_slice) if int(label) in LABEL_DESCRIPTIONS]
    if not labels:
        return ["No predicted tumor region on this slice"]
    return [LABEL_DESCRIPTIONS[label] for label in labels]


def slice_volume(
    base_volume: np.ndarray,
    label_map: np.ndarray,
    plane: str,
    slice_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if plane == "axial":
        base = base_volume[:, :, slice_index]
        label = label_map[:, :, slice_index]
    elif plane == "coronal":
        base = base_volume[:, slice_index, :]
        label = label_map[:, slice_index, :]
    elif plane == "sagittal":
        base = base_volume[slice_index, :, :]
        label = label_map[slice_index, :, :]
    else:
        raise KeyError(f"Unknown plane '{plane}'")
    return np.flipud(base.T), np.flipud(label.T)


def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    data = slice_data.astype(np.float32)
    if np.all(data == 0):
        return np.zeros_like(data, dtype=np.uint8)
    min_value = float(np.min(data))
    max_value = float(np.max(data))
    if max_value - min_value < 1e-6:
        return np.zeros_like(data, dtype=np.uint8)
    return ((data - min_value) / (max_value - min_value) * 255.0).astype(np.uint8)
