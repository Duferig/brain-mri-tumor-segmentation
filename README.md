# Brain MRI Tumor Segmentation

Research prototype for automated brain tumor segmentation on multimodal MRI.
The project explores deep learning methods that can support medical image analysis,
reproducible experiments, and clear visualization of segmentation results.

## Features

- `PyTorch + MONAI` training pipeline for a baseline `3D U-Net` model and an improved `SwinUNETR` model.
- BraTS manifest generation and dataset validation utilities.
- Scripts for model training, evaluation, and experiment comparison.
- `FastAPI` inference service that returns `seg.nii.gz`, preview PNG images, and voxel statistics.
- `Streamlit` fallback demo interface for uploading four MRI modalities and reviewing segmentation results.
- React/Vite web interface for the main interactive demo.

## Project Structure

- `src/brain_mri_segmentation/ml` - data processing, models, training, evaluation, and inference.
- `src/brain_mri_segmentation/api` - HTTP API for model inference and artifact access.
- `src/brain_mri_segmentation/ui` - Streamlit demo UI.
- `web` - React/Vite web interface for the FastAPI backend.
- `configs` - TOML configuration files for training and inference.
- `artifacts` - local model weights, metrics, predictions, and generated runtime artifacts.
- `docs` - Colab workflow and supporting project notes.

## Environment

Python `3.10` to `3.12` is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Quick Start

1. Prepare the BraTS dataset and generate train/validation/test manifests.
2. Train the baseline and improved models.
3. Start the FastAPI inference service.
4. Start the demo UI.

```powershell
brain-seg-manifest --dataset-root D:\data\BraTS --output-dir artifacts\manifests
brain-seg-train --config configs\train_baseline.toml
brain-seg-train --config configs\train_improved.toml
brain-seg-api --config configs\inference.toml
brain-seg-ui --config configs\inference.toml
```

## Web Demo

The main demo UI is located in `web/` and runs on top of the FastAPI backend.

```powershell
brain-seg-api --config configs\inference.toml
cd web
npm install
npm run dev
```

After startup, open `http://127.0.0.1:5173`.

The `brain-seg-ui` Streamlit interface is kept as a fallback option.

## Data Format

Each case is expected to contain four MRI modalities:

- `T1`
- `T1ce`
- `T2`
- `FLAIR`

During training, the label map is converted into the `TC`, `WT`, and `ET` target regions.
The project supports both common BraTS enhancing tumor label variants:

- classic `label 4`;
- BraTS 2023 GLI `label 3`.

Check label distribution in a local dataset with:

```powershell
python scripts\check_labels.py D:\data\BraTS
```

If `ET` is stored as `label 3`, old weights trained with the standard MONAI `label 4`
mapping should be retrained or fine-tuned.

## Validation

```powershell
python -m compileall src tests
pytest
```

If full runtime dependencies are not installed, the static syntax check should still pass.
