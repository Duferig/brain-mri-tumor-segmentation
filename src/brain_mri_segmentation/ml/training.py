from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.utils import set_determinism

from brain_mri_segmentation.ml.brats import load_manifest
from brain_mri_segmentation.ml.config import TrainingAppConfig, load_training_config
from brain_mri_segmentation.ml.metrics import compute_region_metrics, threshold_predictions
from brain_mri_segmentation.ml.models import build_model, load_weights, resolve_device, save_checkpoint
from brain_mri_segmentation.ml.transforms import build_eval_transform, build_train_transform
from brain_mri_segmentation.settings import dump_json


METRIC_KEYS = (
    "dice_tc",
    "dice_wt",
    "dice_et",
    "hd95_tc",
    "hd95_wt",
    "hd95_et",
    "mean_dice",
    "mean_hd95",
)


def build_dataset(
    records: list[dict],
    transform,
    cache_rate: float,
    persistent_cache_dir: Path | None = None,
):
    if persistent_cache_dir is not None:
        return PersistentDataset(
            data=records,
            transform=transform,
            cache_dir=persistent_cache_dir,
            hash_transform=pickle_hashing,
        )
    if cache_rate > 0:
        return CacheDataset(
            data=records,
            transform=transform,
            cache_rate=cache_rate,
            runtime_cache=True,
            progress=False,
        )
    return Dataset(data=records, transform=transform)


def create_loaders(config: TrainingAppConfig) -> tuple[DataLoader, DataLoader]:
    train_records = load_manifest(config.data.train_manifest)
    val_records = load_manifest(config.data.val_manifest)
    train_dataset = build_dataset(
        train_records,
        build_train_transform(config.data.roi_size),
        cache_rate=config.data.cache_rate,
        persistent_cache_dir=(
            config.data.persistent_cache_dir / "train"
            if config.data.persistent_cache_dir is not None
            else None
        ),
    )
    val_dataset = build_dataset(
        val_records,
        build_eval_transform(),
        cache_rate=config.data.cache_rate,
        persistent_cache_dir=(
            config.data.persistent_cache_dir / "val"
            if config.data.persistent_cache_dir is not None
            else None
        ),
    )
    train_loader_kwargs = {
        "batch_size": config.data.batch_size,
        "shuffle": True,
        "num_workers": config.data.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    val_num_workers = (
        config.data.val_num_workers
        if config.data.val_num_workers is not None
        else config.data.num_workers
    )
    val_loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": val_num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if config.data.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = config.data.persistent_workers
        if config.data.prefetch_factor is not None:
            train_loader_kwargs["prefetch_factor"] = config.data.prefetch_factor
    if val_num_workers > 0:
        val_loader_kwargs["persistent_workers"] = (
            config.data.val_persistent_workers
            if config.data.val_persistent_workers is not None
            else config.data.persistent_workers
        )
        val_prefetch_factor = (
            config.data.val_prefetch_factor
            if config.data.val_prefetch_factor is not None
            else config.data.prefetch_factor
        )
        if val_prefetch_factor is not None:
            val_loader_kwargs["prefetch_factor"] = val_prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        **val_loader_kwargs,
    )
    return train_loader, val_loader


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainingAppConfig,
) -> dict[str, float]:
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
            metric_records.append(
                compute_region_metrics(
                    predictions,
                    labels,
                    include_hd95=config.training.compute_hd95,
                )
            )
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


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = _history_fieldnames(history)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(history)


def _history_fieldnames(history: list[dict[str, float]]) -> list[str]:
    fieldnames: list[str] = []
    for record in history:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _find_nvidia_smi() -> str | None:
    candidates = [
        shutil.which("nvidia-smi"),
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def query_gpu_stats(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    nvidia_smi = _find_nvidia_smi()
    if nvidia_smi is None:
        return None
    command = [
        nvidia_smi,
        "-i",
        str(device.index or 0),
        "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip().splitlines()
    if not output:
        return None
    parts = [part.strip() for part in output[0].split(",")]
    if len(parts) != 4:
        return None
    temperature_c, utilization_pct, memory_used_mb, memory_total_mb = (
        int(parts[0]),
        int(parts[1]),
        int(parts[2]),
        int(parts[3]),
    )
    return {
        "temperature_c": temperature_c,
        "utilization_pct": utilization_pct,
        "memory_used_mb": memory_used_mb,
        "memory_total_mb": memory_total_mb,
    }


def enforce_gpu_limits(
    config: TrainingAppConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> None:
    monitor_interval = config.hardware.gpu_monitor_interval_steps
    if device.type != "cuda" or monitor_interval <= 0 or global_step % monitor_interval != 0:
        return
    stats = query_gpu_stats(device)
    if stats is None:
        return
    if (
        config.hardware.max_gpu_temperature_c is not None
        and stats["temperature_c"] > config.hardware.max_gpu_temperature_c
    ):
        raise RuntimeError(
            "Stopping training because GPU temperature exceeded the configured limit: "
            f"{stats['temperature_c']}C > {config.hardware.max_gpu_temperature_c}C "
            f"(epoch={epoch}, step={global_step})."
        )
    if (
        config.hardware.max_gpu_utilization_pct is not None
        and stats["utilization_pct"] > config.hardware.max_gpu_utilization_pct
    ):
        raise RuntimeError(
            "Stopping training because GPU utilization exceeded the configured limit: "
            f"{stats['utilization_pct']}% > {config.hardware.max_gpu_utilization_pct}% "
            f"(epoch={epoch}, step={global_step})."
        )
    if (
        config.hardware.max_gpu_memory_used_mb is not None
        and stats["memory_used_mb"] > config.hardware.max_gpu_memory_used_mb
    ):
        raise RuntimeError(
            "Stopping training because GPU memory usage exceeded the configured limit: "
            f"{stats['memory_used_mb']} MiB > {config.hardware.max_gpu_memory_used_mb} MiB "
            f"(epoch={epoch}, step={global_step})."
        )


def _is_finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def build_loss(config: TrainingAppConfig, device: torch.device) -> torch.nn.Module:
    loss_weight = (
        torch.tensor(config.training.loss_weight, dtype=torch.float32)
        if config.training.loss_weight is not None
        else None
    )
    common_kwargs = {
        "sigmoid": True,
        "squared_pred": True,
        "reduction": "mean",
        "lambda_dice": config.training.loss_lambda_dice,
    }
    if config.training.loss_name == "dice_ce":
        return DiceCELoss(
            **common_kwargs,
            weight=loss_weight,
            lambda_ce=config.training.loss_lambda_other,
        ).to(device)
    if config.training.loss_name == "dice_focal":
        focal_weight = (
            list(config.training.loss_weight)
            if config.training.loss_weight is not None
            else None
        )
        return DiceFocalLoss(
            **common_kwargs,
            gamma=config.training.loss_gamma,
            weight=focal_weight,
            lambda_focal=config.training.loss_lambda_other,
        ).to(device)
    raise ValueError(f"Unknown loss name '{config.training.loss_name}'")


def train(config: TrainingAppConfig) -> dict[str, float]:
    config.experiment.output_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    set_determinism(seed=config.training.seed)
    device = resolve_device(config.hardware.device)
    train_loader, val_loader = create_loaders(config)
    model = build_model(
        model_name=config.training.model_name,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        roi_size=config.data.roi_size,
    )
    if config.training.pretrained_weights is not None:
        load_weights(
            model,
            config.training.pretrained_weights,
            map_location="cpu",
            strict=config.training.pretrained_strict,
        )
        print(f"Loaded pretrained weights from {config.training.pretrained_weights}.")
    model = model.to(device)
    loss_fn = build_loss(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    best_metrics = {"mean_dice": 0.0}
    best_epoch = 0
    history: list[dict[str, float]] = []
    stale_epochs = 0
    global_step = 0

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        skipped_non_finite_batches = 0
        for batch in train_loader:
            global_step += 1
            images = batch["image"].to(device, non_blocking=torch.cuda.is_available())
            labels = batch["label"].to(device, non_blocking=torch.cuda.is_available())
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if not _is_finite_tensor(loss):
                skipped_non_finite_batches += 1
                print(
                    f"warning: skipping non-finite loss at epoch={epoch} step={global_step}"
                )
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            epoch_loss += float(loss.item())
            enforce_gpu_limits(config, device=device, epoch=epoch, global_step=global_step)
        if steps == 0:
            raise RuntimeError(
                f"All training batches were non-finite at epoch {epoch}; stopping training."
            )
        train_loss = epoch_loss / steps
        record: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "skipped_non_finite_batches": float(skipped_non_finite_batches),
            **{key: 0.0 for key in METRIC_KEYS},
        }

        if epoch % config.training.val_interval == 0:
            metrics = validate(model, val_loader, device, config)
            record.update(metrics)
            if metrics["mean_dice"] > best_metrics["mean_dice"]:
                best_metrics = metrics
                best_epoch = epoch
                stale_epochs = 0
                save_checkpoint(
                    path=config.artifacts.checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    metadata={
                        "experiment_name": config.experiment.name,
                        "model_name": config.training.model_name,
                        "config_path": str(config.config_path),
                        "pretrained_weights": (
                            str(config.training.pretrained_weights)
                            if config.training.pretrained_weights is not None
                            else None
                        ),
                        "loss_name": config.training.loss_name,
                    },
                )
            else:
                stale_epochs += 1
        history.append(record)
        write_history(config.artifacts.history_path, history)
        message = (
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"mean_dice={record.get('mean_dice', 0.0):.4f}"
        )
        if skipped_non_finite_batches:
            message += f" skipped_non_finite_batches={skipped_non_finite_batches}"
        print(message)
        if stale_epochs >= config.training.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    summary = {
        "experiment_name": config.experiment.name,
        "model_name": config.training.model_name,
        "best_epoch": best_epoch,
        **best_metrics,
        "checkpoint_path": str(config.artifacts.checkpoint_path),
        "history_path": str(config.artifacts.history_path),
    }
    dump_json(config.artifacts.summary_path, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a BraTS segmentation model.")
    parser.add_argument("--config", required=True, help="Path to training TOML config.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = train(load_training_config(args.config))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
