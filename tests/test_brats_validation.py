from __future__ import annotations

import pytest


np = pytest.importorskip("numpy")
nib = pytest.importorskip("nibabel")

from brain_mri_segmentation.ml.brats import CaseFiles, DataValidationError, validate_case_files  # noqa: E402


def _write_nifti(path, shape) -> None:
    data = np.zeros(shape, dtype=np.float32)
    image = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(image, str(path))


def test_validate_case_files_accepts_matching_modalities(tmp_path) -> None:
    for name in ("t1", "t1ce", "t2", "flair", "seg"):
        _write_nifti(tmp_path / f"case_{name}.nii.gz", (8, 8, 8))
    case = CaseFiles(
        case_id="case",
        t1=tmp_path / "case_t1.nii.gz",
        t1ce=tmp_path / "case_t1ce.nii.gz",
        t2=tmp_path / "case_t2.nii.gz",
        flair=tmp_path / "case_flair.nii.gz",
        label=tmp_path / "case_seg.nii.gz",
    )
    validate_case_files(case)


def test_validate_case_files_rejects_shape_mismatch(tmp_path) -> None:
    _write_nifti(tmp_path / "case_t1.nii.gz", (8, 8, 8))
    _write_nifti(tmp_path / "case_t1ce.nii.gz", (8, 8, 8))
    _write_nifti(tmp_path / "case_t2.nii.gz", (8, 8, 8))
    _write_nifti(tmp_path / "case_flair.nii.gz", (9, 8, 8))
    _write_nifti(tmp_path / "case_seg.nii.gz", (8, 8, 8))
    case = CaseFiles(
        case_id="case",
        t1=tmp_path / "case_t1.nii.gz",
        t1ce=tmp_path / "case_t1ce.nii.gz",
        t2=tmp_path / "case_t2.nii.gz",
        flair=tmp_path / "case_flair.nii.gz",
        label=tmp_path / "case_seg.nii.gz",
    )
    with pytest.raises(DataValidationError):
        validate_case_files(case)

