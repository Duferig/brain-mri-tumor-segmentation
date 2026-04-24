from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from brain_mri_segmentation.settings import ROOT_DIR, load_toml, resolve_path


def _tuple3(values: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {values!r}")
    return int(values[0]), int(values[1]), int(values[2])


def _float_tuple(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    output_dir: Path


@dataclass(slots=True)
class DataConfig:
    train_manifest: Path
    val_manifest: Path
    test_manifest: Path
    num_workers: int
    val_num_workers: int | None
    persistent_workers: bool
    val_persistent_workers: bool | None
    prefetch_factor: int | None
    val_prefetch_factor: int | None
    cache_rate: float
    persistent_cache_dir: Path | None
    roi_size: tuple[int, int, int]
    batch_size: int
    sw_batch_size: int
    infer_overlap: float


@dataclass(slots=True)
class TrainingConfig:
    model_name: str
    seed: int
    epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    amp: bool
    threshold: float
    val_interval: int
    compute_hd95: bool
    pretrained_weights: Path | None = None
    pretrained_strict: bool = True
    loss_name: str = "dice_ce"
    loss_weight: tuple[float, ...] | None = None
    loss_gamma: float = 2.0
    loss_lambda_dice: float = 1.0
    loss_lambda_other: float = 1.0


@dataclass(slots=True)
class ModelConfig:
    in_channels: int
    out_channels: int


@dataclass(slots=True)
class HardwareConfig:
    device: str
    max_gpu_temperature_c: int | None = None
    max_gpu_utilization_pct: int | None = None
    max_gpu_memory_used_mb: int | None = None
    gpu_monitor_interval_steps: int = 0


@dataclass(slots=True)
class ArtifactConfig:
    checkpoint_path: Path
    history_path: Path
    summary_path: Path


@dataclass(slots=True)
class TrainingAppConfig:
    experiment: ExperimentConfig
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig
    hardware: HardwareConfig
    artifacts: ArtifactConfig
    config_path: Path


@dataclass(slots=True)
class InferenceModelConfig:
    key: str
    name: str
    display_name: str
    weights: Path
    roi_size: tuple[int, int, int]
    device: str


@dataclass(slots=True)
class RuntimeConfig:
    artifacts_dir: Path
    predictions_dir: Path
    uploads_dir: Path
    api_host: str
    api_port: int
    preview_modality: str
    preview_opacity: float
    max_upload_size_mb: int


@dataclass(slots=True)
class UIConfig:
    api_url: str
    default_model: str


@dataclass(slots=True)
class InferenceAppConfig:
    runtime: RuntimeConfig
    models: dict[str, InferenceModelConfig]
    ui: UIConfig
    config_path: Path


def load_training_config(path: str | Path) -> TrainingAppConfig:
    config_path = resolve_path(path, ROOT_DIR)
    payload = load_toml(config_path)
    experiment = payload["experiment"]
    data = payload["data"]
    training = payload["training"]
    model = payload["model"]
    hardware = payload["hardware"]
    artifacts = payload["artifacts"]
    return TrainingAppConfig(
        experiment=ExperimentConfig(
            name=experiment["name"],
            output_dir=resolve_path(experiment["output_dir"]),
        ),
        data=DataConfig(
            train_manifest=resolve_path(data["train_manifest"]),
            val_manifest=resolve_path(data["val_manifest"]),
            test_manifest=resolve_path(data["test_manifest"]),
            num_workers=int(data["num_workers"]),
            val_num_workers=(
                int(data["val_num_workers"])
                if data.get("val_num_workers") is not None
                else None
            ),
            persistent_workers=bool(data.get("persistent_workers", False)),
            val_persistent_workers=(
                bool(data["val_persistent_workers"])
                if data.get("val_persistent_workers") is not None
                else None
            ),
            prefetch_factor=(
                int(data["prefetch_factor"])
                if data.get("prefetch_factor") is not None
                else None
            ),
            val_prefetch_factor=(
                int(data["val_prefetch_factor"])
                if data.get("val_prefetch_factor") is not None
                else None
            ),
            cache_rate=float(data["cache_rate"]),
            persistent_cache_dir=(
                resolve_path(data["persistent_cache_dir"])
                if data.get("persistent_cache_dir") is not None
                else None
            ),
            roi_size=_tuple3(data["roi_size"]),
            batch_size=int(data["batch_size"]),
            sw_batch_size=int(data["sw_batch_size"]),
            infer_overlap=float(data["infer_overlap"]),
        ),
        training=TrainingConfig(
            model_name=training["model_name"],
            seed=int(training["seed"]),
            epochs=int(training["epochs"]),
            patience=int(training["patience"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            amp=bool(training["amp"]),
            threshold=float(training["threshold"]),
            val_interval=int(training["val_interval"]),
            compute_hd95=bool(training.get("compute_hd95", True)),
            pretrained_weights=(
                resolve_path(training["pretrained_weights"])
                if training.get("pretrained_weights") is not None
                else None
            ),
            pretrained_strict=bool(training.get("pretrained_strict", True)),
            loss_name=str(training.get("loss_name", "dice_ce")),
            loss_weight=(
                _float_tuple(training["loss_weight"])
                if training.get("loss_weight") is not None
                else None
            ),
            loss_gamma=float(training.get("loss_gamma", 2.0)),
            loss_lambda_dice=float(training.get("loss_lambda_dice", 1.0)),
            loss_lambda_other=float(training.get("loss_lambda_other", 1.0)),
        ),
        model=ModelConfig(
            in_channels=int(model["in_channels"]),
            out_channels=int(model["out_channels"]),
        ),
        hardware=HardwareConfig(
            device=hardware["device"],
            max_gpu_temperature_c=(
                int(hardware["max_gpu_temperature_c"])
                if hardware.get("max_gpu_temperature_c") is not None
                else None
            ),
            max_gpu_utilization_pct=(
                int(hardware["max_gpu_utilization_pct"])
                if hardware.get("max_gpu_utilization_pct") is not None
                else None
            ),
            max_gpu_memory_used_mb=(
                int(hardware["max_gpu_memory_used_mb"])
                if hardware.get("max_gpu_memory_used_mb") is not None
                else None
            ),
            gpu_monitor_interval_steps=int(hardware.get("gpu_monitor_interval_steps", 0)),
        ),
        artifacts=ArtifactConfig(
            checkpoint_path=resolve_path(artifacts["checkpoint_path"]),
            history_path=resolve_path(artifacts["history_path"]),
            summary_path=resolve_path(artifacts["summary_path"]),
        ),
        config_path=config_path,
    )


def load_inference_config(path: str | Path) -> InferenceAppConfig:
    config_path = resolve_path(path, ROOT_DIR)
    payload = load_toml(config_path)
    runtime = payload["runtime"]
    models = payload["models"]
    ui = payload["ui"]
    return InferenceAppConfig(
        runtime=RuntimeConfig(
            artifacts_dir=resolve_path(runtime["artifacts_dir"]),
            predictions_dir=resolve_path(runtime["predictions_dir"]),
            uploads_dir=resolve_path(runtime["uploads_dir"]),
            api_host=str(runtime["api_host"]),
            api_port=int(runtime["api_port"]),
            preview_modality=str(runtime["preview_modality"]),
            preview_opacity=float(runtime["preview_opacity"]),
            max_upload_size_mb=int(runtime["max_upload_size_mb"]),
        ),
        models={
            key: InferenceModelConfig(
                key=key,
                name=value["name"],
                display_name=str(value.get("display_name", key)),
                weights=resolve_path(value["weights"]),
                roi_size=_tuple3(value["roi_size"]),
                device=str(value["device"]),
            )
            for key, value in models.items()
        },
        ui=UIConfig(
            api_url=str(ui["api_url"]).rstrip("/"),
            default_model=str(ui["default_model"]),
        ),
        config_path=config_path,
    )
