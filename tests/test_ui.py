from __future__ import annotations

import gzip

import nibabel as nib
import numpy as np

from brain_mri_segmentation.ui.app import compare_prediction_to_reference


def _label_bytes(label: np.ndarray) -> bytes:
    image = nib.Nifti1Image(label.astype(np.uint8), affine=np.eye(4))
    return image.to_bytes()


def test_compare_prediction_to_reference_returns_perfect_dice_for_identical_labels() -> None:
    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[2:5, 2:5, 2:5] = 1
    label[3:4, 3:4, 3:4] = 4

    metrics = compare_prediction_to_reference(_label_bytes(label), _label_bytes(label))

    assert metrics["mean_dice"] == 1.0
    assert metrics["dice_tc"] == 1.0
    assert metrics["dice_wt"] == 1.0
    assert metrics["dice_et"] == 1.0


def test_compare_prediction_to_reference_accepts_gzipped_nifti_payloads() -> None:
    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[2:5, 2:5, 2:5] = 2
    label[3:5, 3:5, 3:5] = 1
    label[4:5, 4:5, 4:5] = 4

    payload = gzip.compress(_label_bytes(label))
    metrics = compare_prediction_to_reference(payload, payload)

    assert metrics["mean_dice"] == 1.0
    assert metrics["dice_tc"] == 1.0
    assert metrics["dice_wt"] == 1.0
    assert metrics["dice_et"] == 1.0


def test_compare_prediction_to_reference_rejects_shape_mismatch() -> None:
    prediction = np.zeros((8, 8, 8), dtype=np.uint8)
    reference = np.zeros((6, 6, 6), dtype=np.uint8)

    try:
        compare_prediction_to_reference(_label_bytes(prediction), _label_bytes(reference))
    except ValueError as error:
        assert "shapes do not match" in str(error)
    else:
        raise AssertionError("Expected ValueError for mismatched shapes")
