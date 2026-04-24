from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count segmentation label values in a BraTS-like dataset."
    )
    parser.add_argument("dataset_root", type=Path, help="Path to the dataset root directory.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of segmentation files to inspect. Use 0 for all files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import nibabel as nib
    import numpy as np

    files = sorted(args.dataset_root.rglob("*seg*.nii*"))
    inspected = files if args.limit == 0 else files[: args.limit]

    print(f"Found seg files: {len(files)}")
    print(f"Inspecting seg files: {len(inspected)}")

    total: dict[int | str, int] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, "other": 0}

    for path in inspected:
        data = np.asarray(nib.load(str(path)).get_fdata()).round().astype(np.int16)
        values, counts = np.unique(data, return_counts=True)

        for value, count in zip(values, counts):
            label = int(value)
            if label in total:
                total[label] += int(count)
            else:
                total["other"] += int(count)

    print("\nVoxel counts:")
    for label, count in total.items():
        print(f"label {label}: {count:,}")


if __name__ == "__main__":
    main()
