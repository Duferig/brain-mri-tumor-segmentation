from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES = ("t1", "t1ce", "t2", "flair")


def create_case(case_dir: Path, seed: int, shape: tuple[int, int, int]) -> None:
    rng = np.random.default_rng(seed)
    case_dir.mkdir(parents=True, exist_ok=True)
    zz, yy, xx = np.indices(shape)
    center = np.array(shape) / 2 + rng.integers(-2, 3, size=3)
    dist = np.sqrt(((zz - center[0]) ** 2) + ((yy - center[1]) ** 2) + ((xx - center[2]) ** 2))

    edema = dist <= rng.integers(10, 13)
    tumor_core = dist <= rng.integers(6, 9)
    enhancing = dist <= rng.integers(3, 5)

    label = np.zeros(shape, dtype=np.uint8)
    label[edema] = 2
    label[tumor_core] = 1
    label[enhancing] = 4

    affine = np.eye(4, dtype=np.float32)
    for index, modality in enumerate(MODALITIES, start=1):
        volume = rng.normal(loc=0.0, scale=0.05, size=shape).astype(np.float32)
        volume += (edema.astype(np.float32) * (0.15 * index))
        volume += (tumor_core.astype(np.float32) * (0.3 * index))
        volume += (enhancing.astype(np.float32) * (0.45 * index))
        image = nib.Nifti1Image(volume, affine=affine)
        nib.save(image, str(case_dir / f"{case_dir.name}_{modality}.nii.gz"))

    label_img = nib.Nifti1Image(label.astype(np.uint8), affine=affine)
    nib.save(label_img, str(case_dir / f"{case_dir.name}_seg.nii.gz"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic BraTS-like dataset.")
    parser.add_argument("--output-dir", default="artifacts/smoke_dataset")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--size", type=int, default=32)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (args.size, args.size, args.size)
    for case_index in range(args.cases):
        create_case(output_dir / f"BraTS-SYN-{case_index:03d}", seed=case_index + 1, shape=shape)
    print(output_dir)


if __name__ == "__main__":
    main()
