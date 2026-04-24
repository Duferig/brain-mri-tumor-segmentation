from __future__ import annotations

import argparse
import json
import time

from brain_mri_segmentation.ml.config import load_training_config
from brain_mri_segmentation.ml.training import create_loaders


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test MONAI DataLoader settings on Windows."
    )
    parser.add_argument("--config", required=True, help="Path to training TOML config.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers.")
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override persistent_workers for the smoke test.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Override prefetch_factor for the smoke test.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override training batch_size."
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=2,
        help="How many train batches to iterate before exiting.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_training_config(args.config)
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.prefetch_factor is not None:
        config.data.prefetch_factor = args.prefetch_factor
    if config.data.num_workers > 0:
        if args.persistent_workers is not None:
            config.data.persistent_workers = args.persistent_workers
    else:
        config.data.persistent_workers = False
        config.data.prefetch_factor = None

    started_at = time.perf_counter()
    train_loader, _ = create_loaders(config)
    loader_ready_at = time.perf_counter()

    batch_shapes: list[dict[str, object]] = []
    for index, batch in enumerate(train_loader, start=1):
        batch_shapes.append(
            {
                "batch_index": index,
                "image_shape": list(batch["image"].shape),
                "label_shape": list(batch["label"].shape),
            }
        )
        if index >= args.batches:
            break

    finished_at = time.perf_counter()
    print(
        json.dumps(
            {
                "config": str(config.config_path),
                "num_workers": config.data.num_workers,
                "persistent_workers": config.data.persistent_workers,
                "prefetch_factor": config.data.prefetch_factor,
                "batch_size": config.data.batch_size,
                "loader_init_s": round(loader_ready_at - started_at, 3),
                "iterate_s": round(finished_at - loader_ready_at, 3),
                "batches_checked": len(batch_shapes),
                "batch_shapes": batch_shapes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
