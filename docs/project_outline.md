# Project Outline

## 1. Domain Analysis

- Clinical motivation for brain tumor detection and segmentation.
- MRI specifics and multimodal input data.
- U-Net, attention-based, and transformer-like segmentation architectures.
- BraTS dataset format and target regions.

## 2. Problem Statement And Requirements

- Project goal and quality criteria.
- Expected input and output artifacts.
- Architecture: training pipeline, inference API, and web demo.
- Practical limits of the research prototype.

## 3. Implementation

- Manifest generation and data validation.
- Preprocessing and augmentation.
- Baseline `3D U-Net`.
- Improved `SwinUNETR`.
- FastAPI service and React/Streamlit interfaces.

## 4. Experiments

- Train/validation/test split design.
- Dice scores for `WT`, `TC`, and `ET`.
- `HD95` and supporting quality indicators.
- Baseline and improved model comparison.
- Error analysis, edge cases, and future improvement directions.

