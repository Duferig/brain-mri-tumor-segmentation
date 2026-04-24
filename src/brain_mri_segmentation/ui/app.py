from __future__ import annotations

import argparse
import gzip
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import requests
import streamlit as st
import torch
from streamlit.web import cli as stcli

from brain_mri_segmentation.ml.labels import brats_label_to_regions
from brain_mri_segmentation.ml.metrics import compute_region_metrics
from brain_mri_segmentation.ml.config import load_inference_config


def _config_path() -> str:
    return os.getenv("BRAIN_SEG_INFERENCE_CONFIG", "configs/inference.toml")


def render_app(config_path: str | None = None) -> None:
    config = load_inference_config(config_path or _config_path())
    st.set_page_config(page_title="Brain MRI Segmentation", layout="wide")
    st.title("Automatic Brain Tumor Segmentation on MRI")
    st.caption("Загрузите четыре MRI-модальности и выполните инференс через FastAPI сервис.")

    with st.sidebar:
        api_url = st.text_input("API URL", value=config.ui.api_url)
        model = st.selectbox(
            "Model",
            options=sorted(config.models.keys()),
            index=_default_index(config),
            format_func=lambda key: config.models[key].display_name,
        )
        st.markdown("Ожидаются NIfTI файлы `.nii` или `.nii.gz`.")

    columns = st.columns(4)
    uploads = {
        "t1": columns[0].file_uploader("T1", type=["nii", "gz"]),
        "t1ce": columns[1].file_uploader("T1ce", type=["nii", "gz"]),
        "t2": columns[2].file_uploader("T2", type=["nii", "gz"]),
        "flair": columns[3].file_uploader("FLAIR", type=["nii", "gz"]),
    }
    reference_seg = st.file_uploader(
        "Reference Segmentation (optional)",
        type=["nii", "gz"],
        help="Upload a ground-truth seg.nii.gz to compare the prediction against the target.",
    )

    if st.button("Run Segmentation", type="primary"):
        missing = [name for name, file_obj in uploads.items() if file_obj is None]
        if missing:
            st.error(f"Не загружены обязательные модальности: {', '.join(missing)}")
        else:
            with st.spinner("Выполняется инференс..."):
                response = call_predict(api_url, uploads, model)
            if "error" in response:
                st.error(response["error"])
            else:
                st.session_state["prediction_response"] = response

    payload = st.session_state.get("prediction_response")
    if payload:
        model_key = payload["model_used"]
        model_label = config.models[model_key].display_name if model_key in config.models else model_key
        st.subheader("Prediction")
        info_cols = st.columns(3)
        info_cols[0].metric("Prediction ID", payload["prediction_id"])
        info_cols[1].metric("Model", model_label)
        info_cols[2].metric("WT voxels", payload["voxel_statistics"].get("WT", 0))
        st.json(payload["voxel_statistics"])
        st.markdown(
            (
                "**Color legend:** yellow = label 1 / TC (non-enhancing core), "
                "blue = label 2 / edema-only WT, red = label 4 / ET (enhancing tumor)."
            )
        )

        for image_info in payload["preview_images"]:
            st.markdown(f"### {image_info['plane'].title()} Slice {image_info['slice_index']}")
            preview_modality = image_info.get("modality", config.runtime.preview_modality.upper())
            original_url = image_info.get("original_url") or image_info.get("url")
            overlay_url = image_info.get("overlay_url") or image_info.get("url")
            highlighted_labels = image_info.get("highlighted_labels", [])
            st.caption(f"Preview modality: {preview_modality}")
            if original_url == overlay_url:
                st.warning(
                    "Original and overlay previews are identical. This usually means the UI is "
                    "showing a legacy API response cached before the preview update. Restart the "
                    "API, rerun segmentation, and refresh the page."
                )
            preview_cols = st.columns(2)
            preview_cols[0].image(
                original_url,
                caption="Original MRI slice",
                width="stretch",
            )
            preview_cols[1].image(
                overlay_url,
                caption="Overlay with tumor mask and contour",
                width="stretch",
            )
            if highlighted_labels:
                st.markdown("**Highlighted on this slice:**")
                for label in highlighted_labels:
                    st.markdown(f"- {label}")

        segmentation_bytes = fetch_binary(payload["segmentation_url"])
        if reference_seg is not None:
            if segmentation_bytes is None:
                st.warning("Could not download the predicted seg.nii.gz for comparison.")
            else:
                try:
                    comparison = compare_prediction_to_reference(
                        prediction_bytes=segmentation_bytes,
                        reference_bytes=reference_seg.getvalue(),
                    )
                except ValueError as error:
                    st.error(f"Reference comparison failed: {error}")
                else:
                    st.subheader("Reference Comparison")
                    st.caption("Dice against the uploaded target seg.nii.gz")
                    compare_cols = st.columns(4)
                    compare_cols[0].metric("Mean Dice", f"{comparison['mean_dice']:.4f}")
                    compare_cols[1].metric("Dice WT", f"{comparison['dice_wt']:.4f}")
                    compare_cols[2].metric("Dice TC", f"{comparison['dice_tc']:.4f}")
                    compare_cols[3].metric("Dice ET", f"{comparison['dice_et']:.4f}")
        if segmentation_bytes is not None:
            st.download_button(
                label="Download seg.nii.gz",
                data=segmentation_bytes,
                file_name="seg.nii.gz",
                mime="application/gzip",
            )
        else:
            st.link_button("Open segmentation", payload["segmentation_url"])


def call_predict(api_url: str, uploads: dict[str, object], model: str) -> dict:
    files = {}
    try:
        for key, file_obj in uploads.items():
            file_obj.seek(0)
            files[key] = (
                file_obj.name,
                file_obj.getvalue(),
                "application/octet-stream",
            )
        response = requests.post(
            f"{api_url.rstrip('/')}/predict",
            files=files,
            data={"model": model},
            timeout=600,
        )
        if response.ok:
            return response.json()
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:  # noqa: BLE001
            error_detail = response.text
        return {"error": f"API error {response.status_code}: {error_detail}"}
    except requests.RequestException as error:
        return {"error": f"Request failed: {error}"}


def fetch_binary(url: str) -> bytes | None:
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException:
        return None
    return response.content


def compare_prediction_to_reference(
    prediction_bytes: bytes,
    reference_bytes: bytes,
) -> dict[str, float]:
    prediction_label = _load_label_from_bytes(prediction_bytes)
    reference_label = _load_label_from_bytes(reference_bytes)
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


def _load_label_from_bytes(payload: bytes) -> np.ndarray:
    if payload[:2] == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    image = nib.Nifti1Image.from_bytes(payload)
    return np.asarray(image.get_fdata(dtype=np.float32)).round().astype(np.uint8)


def _default_index(config) -> int:
    keys = sorted(config.models.keys())
    if config.ui.default_model in keys:
        return keys.index(config.ui.default_model)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the Streamlit demo app.")
    parser.add_argument("--config", default=_config_path(), help="Path to inference TOML config.")
    return parser


def launch() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    os.environ["BRAIN_SEG_INFERENCE_CONFIG"] = args.config
    script_path = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", script_path, "--server.headless=true"]
    raise SystemExit(stcli.main())


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
