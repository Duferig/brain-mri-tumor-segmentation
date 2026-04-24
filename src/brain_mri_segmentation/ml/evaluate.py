from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference

from brain_mri_segmentation.ml.brats import load_manifest
from brain_mri_segmentation.ml.config import TrainingAppConfig, load_training_config
from brain_mri_segmentation.ml.metrics import compute_region_metrics, threshold_predictions
from brain_mri_segmentation.ml.models import build_model, load_weights, resolve_device
from brain_mri_segmentation.ml.transforms import build_eval_transform
from brain_mri_segmentation.settings import dump_json, resolve_path


def evaluate_config(config: TrainingAppConfig) -> dict[str, float | str]:
    records = load_manifest(config.data.test_manifest)
    dataset = Dataset(data=records, transform=build_eval_transform())
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = resolve_device(config.hardware.device)
    model = build_model(
        model_name=config.training.model_name,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        roi_size=config.data.roi_size,
    ).to(device)
    load_weights(model, config.artifacts.checkpoint_path, map_location=device)
    model.eval()
    metric_records: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = sliding_window_inference(
                inputs=images,
                roi_size=config.data.roi_size,
                sw_batch_size=config.data.sw_batch_size,
                predictor=model,
                overlap=config.data.infer_overlap,
            )
            predictions = threshold_predictions(logits, threshold=config.training.threshold)
            metric_records.append(compute_region_metrics(predictions, labels))
    summary = {
        "experiment_name": config.experiment.name,
        "model_name": config.training.model_name,
        "checkpoint_path": str(config.artifacts.checkpoint_path),
        **average_records(metric_records),
    }
    summary_path = config.artifacts.summary_path.with_name(
        f"{config.artifacts.summary_path.stem}_test.json"
    )
    dump_json(summary_path, summary)
    return summary


def average_records(metric_records: list[dict[str, float]]) -> dict[str, float]:
    if not metric_records:
        return {
            "dice_tc": 0.0,
            "dice_wt": 0.0,
            "dice_et": 0.0,
            "hd95_tc": 0.0,
            "hd95_wt": 0.0,
            "hd95_et": 0.0,
            "mean_dice": 0.0,
            "mean_hd95": 0.0,
        }
    return {
        key: float(sum(record[key] for record in metric_records) / len(metric_records))
        for key in metric_records[0]
    }


def write_comparison(outputs: list[dict[str, float | str]], comparison_path: Path) -> None:
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(outputs[0].keys()) if outputs else []
    with comparison_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)
    markdown_path = comparison_path.with_suffix(".md")
    lines = [
        "| Experiment | Mean Dice | Dice WT | Dice TC | Dice ET | HD95 WT | HD95 TC | HD95 ET |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for output in outputs:
        lines.append(
            "| {experiment_name} | {mean_dice:.4f} | {dice_wt:.4f} | {dice_tc:.4f} | {dice_et:.4f} | "
            "{hd95_wt:.4f} | {hd95_tc:.4f} | {hd95_et:.4f} |".format(**output)
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one or more trained BraTS models.")
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Training TOML config. Repeat the option to compare multiple models.",
    )
    parser.add_argument(
        "--comparison-path",
        default="artifacts/metrics/model_comparison.csv",
        help="Where to write the combined comparison table.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    outputs = [evaluate_config(load_training_config(config_path)) for config_path in args.config]
    if len(outputs) > 1:
        write_comparison(outputs, resolve_path(args.comparison_path))
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
