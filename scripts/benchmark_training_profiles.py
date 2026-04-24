from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import torch
from monai.losses import DiceCELoss

from brain_mri_segmentation.ml.config import TrainingAppConfig, load_training_config
from brain_mri_segmentation.ml.models import build_model, resolve_device
from brain_mri_segmentation.ml.training import create_loaders


ROOT = Path(__file__).resolve().parents[1]


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.dwLength = ctypes.sizeof(self)


def system_memory_snapshot() -> dict[str, float]:
    status = MEMORYSTATUSEX()
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    total_gb = status.ullTotalPhys / (1024**3)
    available_gb = status.ullAvailPhys / (1024**3)
    used_gb = total_gb - available_gb
    return {
        "system_memory_load_pct": float(status.dwMemoryLoad),
        "system_memory_used_gb": round(used_gb, 2),
        "system_memory_total_gb": round(total_gb, 2),
    }


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss_fn: DiceCELoss,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    amp_enabled: bool,
) -> float:
    images = batch["image"].to(device, non_blocking=device.type == "cuda")
    labels = batch["label"].to(device, non_blocking=device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        logits = model(images)
        loss = loss_fn(logits, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(loss.item())


def run_child_benchmark(
    config_path: str,
    batch_size: int,
    sw_batch_size: int,
    num_workers: int,
    persistent_workers: bool,
    prefetch_factor: int | None,
    cache_rate: float,
    measured_batches: int,
    warmup_batches: int,
    candidate_name: str,
) -> dict[str, Any]:
    config = load_training_config(config_path)
    config = replace(
        config,
        data=replace(
            config.data,
            batch_size=batch_size,
            sw_batch_size=sw_batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            cache_rate=cache_rate,
        ),
    )
    device = resolve_device(config.hardware.device)
    if device.type != "cuda":
        raise RuntimeError("Benchmark is intended for CUDA runs.")

    train_loader, _ = create_loaders(config)
    model = build_model(
        model_name=config.training.model_name,
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        roi_size=config.data.roi_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    amp_enabled = config.training.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    model.train()

    load_times: list[float] = []
    step_times: list[float] = []
    losses: list[float] = []
    peak_memory_mb = 0.0
    peak_system_load_pct = 0.0
    peak_system_used_gb = 0.0

    iterator = iter(train_loader)
    total_batches = warmup_batches + measured_batches

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for batch_idx in range(total_batches):
        t0 = time.perf_counter()
        batch = next(iterator)
        sync_if_cuda(device)
        t1 = time.perf_counter()
        loss_value = train_step(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loss_fn=loss_fn,
            batch=batch,
            device=device,
            amp_enabled=amp_enabled,
        )
        sync_if_cuda(device)
        t2 = time.perf_counter()
        memory = system_memory_snapshot()
        peak_system_load_pct = max(peak_system_load_pct, memory["system_memory_load_pct"])
        peak_system_used_gb = max(peak_system_used_gb, memory["system_memory_used_gb"])
        if device.type == "cuda":
            peak_memory_mb = max(
                peak_memory_mb,
                torch.cuda.max_memory_allocated(device) / (1024**2),
            )
        if batch_idx >= warmup_batches:
            load_times.append(t1 - t0)
            step_times.append(t2 - t1)
            losses.append(loss_value)

    avg_load_s = statistics.mean(load_times)
    avg_step_s = statistics.mean(step_times)
    avg_total_s = avg_load_s + avg_step_s
    samples_per_second = batch_size / avg_total_s
    result = {
        "candidate": candidate_name,
        "success": True,
        "batch_size": batch_size,
        "sw_batch_size": sw_batch_size,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "cache_rate": cache_rate,
        "avg_load_s": round(avg_load_s, 4),
        "avg_step_s": round(avg_step_s, 4),
        "avg_total_batch_s": round(avg_total_s, 4),
        "samples_per_second": round(samples_per_second, 4),
        "peak_gpu_allocated_mb": round(peak_memory_mb, 1),
        "peak_system_memory_load_pct": round(peak_system_load_pct, 1),
        "peak_system_memory_used_gb": round(peak_system_used_gb, 2),
        "mean_loss": round(statistics.mean(losses), 4),
    }
    return result


def run_parent(args: argparse.Namespace) -> int:
    candidates = [
        {
            "candidate_name": "stable-current",
            "batch_size": 2,
            "sw_batch_size": 2,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "cache_rate": 0.0,
        },
        {
            "candidate_name": "workers-2",
            "batch_size": 2,
            "sw_batch_size": 2,
            "num_workers": 2,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "cache_rate": 0.0,
        },
        {
            "candidate_name": "workers-4",
            "batch_size": 2,
            "sw_batch_size": 2,
            "num_workers": 4,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "cache_rate": 0.0,
        },
        {
            "candidate_name": "faster-val",
            "batch_size": 2,
            "sw_batch_size": 4,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "cache_rate": 0.0,
        },
        {
            "candidate_name": "aggressive-batch-3",
            "batch_size": 3,
            "sw_batch_size": 2,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "cache_rate": 0.0,
        },
    ]
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--config",
            args.config,
            "--measured-batches",
            str(args.measured_batches),
            "--warmup-batches",
            str(args.warmup_batches),
        ]
        for key, value in candidate.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                command.extend([flag, json.dumps(value)])
            elif value is None:
                command.extend([flag, "null"])
            else:
                command.extend([flag, str(value)])
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        print(f"\n=== Running {candidate['candidate_name']} ===")
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.returncode != 0:
            failure = {
                "candidate": candidate["candidate_name"],
                "success": False,
                "error": result.stderr.strip() or result.stdout.strip() or "unknown error",
            }
            results.append(failure)
            continue
        child_payload = json.loads(result.stdout.strip().splitlines()[-1])
        results.append(child_payload)

    print("\n=== Summary ===")
    for item in results:
        if item["success"]:
            print(
                f"{item['candidate']}: {item['samples_per_second']} samples/s, "
                f"{item['avg_total_batch_s']} s/batch, "
                f"GPU peak {item['peak_gpu_allocated_mb']} MiB, "
                f"RAM peak {item['peak_system_memory_load_pct']}%"
            )
        else:
            print(f"{item['candidate']}: FAILED -> {item['error']}")

    best = [
        item
        for item in results
        if item["success"] and item["peak_system_memory_load_pct"] < 90.0
    ]
    if best:
        best = sorted(best, key=lambda item: item["samples_per_second"], reverse=True)
        print("\n=== Best Safe Candidate ===")
        print(json.dumps(best[0], indent=2))
    else:
        print("\nNo candidate completed within the safety envelope.")
    return 0


def parse_json_like_bool(value: str) -> bool:
    return json.loads(value.lower())


def parse_prefetch(value: str) -> int | None:
    return None if value == "null" else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark short training profiles on the local GPU.")
    parser.add_argument("--config", default="configs/train_baseline_3060_12gb.toml")
    parser.add_argument("--measured-batches", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--candidate-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sw-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--persistent-workers", type=parse_json_like_bool)
    parser.add_argument("--prefetch-factor", type=parse_prefetch)
    parser.add_argument("--cache-rate", type=float, default=0.0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.child:
        return run_parent(args)
    try:
        result = run_child_benchmark(
            config_path=args.config,
            batch_size=args.batch_size,
            sw_batch_size=args.sw_batch_size,
            num_workers=args.num_workers,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            cache_rate=args.cache_rate,
            measured_batches=args.measured_batches,
            warmup_batches=args.warmup_batches,
            candidate_name=args.candidate_name,
        )
        print(json.dumps(result))
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
