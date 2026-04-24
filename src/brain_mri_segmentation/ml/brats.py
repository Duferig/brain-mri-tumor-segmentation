from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib

from brain_mri_segmentation.ml.labels import MODALITY_ORDER
from brain_mri_segmentation.settings import ROOT_DIR, dump_json, resolve_path


NIFTI_SUFFIXES = (".nii", ".nii.gz")
T1CE_PATTERN = re.compile(r"(t1ce|t1c|t1gd)", re.IGNORECASE)
T1_PATTERN = re.compile(r"(?:^|[_-])t1(?:n)?(?:$|[_-])", re.IGNORECASE)
T2_PATTERN = re.compile(r"(?:^|[_-])t2(?:w)?(?:$|[_-])", re.IGNORECASE)
FLAIR_PATTERN = re.compile(r"(flair|t2f)", re.IGNORECASE)
LABEL_PATTERN = re.compile(r"(seg|label|mask)", re.IGNORECASE)


class DataValidationError(RuntimeError):
    """Raised when a BraTS case is incomplete or malformed."""


@dataclass(slots=True)
class CaseFiles:
    case_id: str
    t1: Path
    t1ce: Path
    t2: Path
    flair: Path
    label: Path | None = None

    def to_manifest_record(self) -> dict[str, Any]:
        record = {
            "case_id": self.case_id,
            "image": [str(self.t1), str(self.t1ce), str(self.t2), str(self.flair)],
            "modality_map": {
                "t1": str(self.t1),
                "t1ce": str(self.t1ce),
                "t2": str(self.t2),
                "flair": str(self.flair),
            },
        }
        if self.label is not None:
            record["label"] = str(self.label)
        return record


def is_nifti(path: Path) -> bool:
    return any(str(path).lower().endswith(suffix) for suffix in NIFTI_SUFFIXES)


def _canonical_name(path: Path) -> str:
    name = path.name.lower()
    for suffix in NIFTI_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _match_modality(files: list[Path], modality: str) -> Path | None:
    patterns = {
        "t1": T1_PATTERN,
        "t1ce": T1CE_PATTERN,
        "t2": T2_PATTERN,
        "flair": FLAIR_PATTERN,
        "label": LABEL_PATTERN,
    }
    candidates: list[Path] = []
    for path in files:
        stem = _canonical_name(path)
        if modality == "t1" and T1CE_PATTERN.search(stem):
            continue
        if patterns[modality].search(stem):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates)[0]


def discover_case_directories(dataset_root: Path) -> list[Path]:
    nifti_dirs = {path.parent for path in dataset_root.rglob("*") if path.is_file() and is_nifti(path)}
    return sorted(nifti_dirs)


def build_case_files(case_dir: Path, require_label: bool = True) -> CaseFiles:
    files = [path for path in case_dir.iterdir() if path.is_file() and is_nifti(path)]
    if not files:
        raise DataValidationError(f"No NIfTI files found in {case_dir}")
    matched = {modality: _match_modality(files, modality) for modality in (*MODALITY_ORDER, "label")}
    missing = [modality for modality in MODALITY_ORDER if matched[modality] is None]
    if missing:
        raise DataValidationError(f"Case {case_dir.name} is missing modalities: {', '.join(missing)}")
    if require_label and matched["label"] is None:
        raise DataValidationError(f"Case {case_dir.name} is missing a segmentation label")
    case = CaseFiles(
        case_id=case_dir.name,
        t1=matched["t1"],
        t1ce=matched["t1ce"],
        t2=matched["t2"],
        flair=matched["flair"],
        label=matched["label"],
    )
    validate_case_files(case, require_label=require_label)
    return case


def validate_case_files(case: CaseFiles, require_label: bool = True) -> None:
    modalities = [case.t1, case.t1ce, case.t2, case.flair]
    for path in modalities:
        if not is_nifti(path):
            raise DataValidationError(f"{path} does not have .nii or .nii.gz extension")
    if require_label and case.label is None:
        raise DataValidationError(f"{case.case_id} is missing label path")
    shapes = {}
    for path in modalities + ([case.label] if case.label else []):
        try:
            image = nib.load(str(path))
        except Exception as error:  # noqa: BLE001
            raise DataValidationError(f"Failed to read NIfTI file {path}: {error}") from error
        shapes[path.name] = image.shape
    unique_shapes = {shape for shape in shapes.values()}
    if len(unique_shapes) != 1:
        raise DataValidationError(
            f"Case {case.case_id} has inconsistent shapes: "
            + ", ".join(f"{name}={shape}" for name, shape in shapes.items())
        )


def discover_cases(dataset_root: str | Path, require_label: bool = True) -> list[CaseFiles]:
    root = resolve_path(dataset_root, ROOT_DIR)
    cases: list[CaseFiles] = []
    for case_dir in discover_case_directories(root):
        cases.append(build_case_files(case_dir, require_label=require_label))
    if not cases:
        raise DataValidationError(f"No valid BraTS cases found under {root}")
    return cases


def save_manifest(records: list[dict[str, Any]], path: str | Path) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    resolved = resolve_path(path)
    return json.loads(resolved.read_text(encoding="utf-8"))


def split_cases(
    cases: list[CaseFiles],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[CaseFiles]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    rng = random.Random(seed)
    shuffled = cases[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_manifests(
    dataset_root: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, Path]:
    cases = discover_cases(dataset_root, require_label=True)
    splits = split_cases(cases, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split_name, split_cases_list in splits.items():
        manifest_path = output / f"{split_name}.json"
        save_manifest([case.to_manifest_record() for case in split_cases_list], manifest_path)
        paths[split_name] = manifest_path
    summary = {
        "dataset_root": str(resolve_path(dataset_root)),
        "seed": seed,
        "counts": {split_name: len(split_cases_list) for split_name, split_cases_list in splits.items()},
    }
    dump_json(output / "summary.json", summary)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create BraTS train/val/test manifests.")
    parser.add_argument("--dataset-root", required=True, help="Path to the BraTS dataset root.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/manifests",
        help="Where to write manifest JSON files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    paths = write_manifests(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    for split_name, manifest_path in paths.items():
        print(f"{split_name}: {manifest_path}")


if __name__ == "__main__":
    main()
