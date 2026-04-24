from __future__ import annotations

import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "artifacts" / "metrics"
OUTPUT_DIR = ROOT / "output" / "diploma" / "assets"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_pipeline_svg()
    write_architecture_svg()
    write_inference_svg()
    write_model_comparison_svg(load_summaries())
    write_training_curve_svg(load_history("baseline_3060_12gb_quality_history.csv"))


def write_pipeline_svg() -> None:
    steps = [
        ("BraTS", "T1 / T1ce / T2 / FLAIR"),
        ("Валидация", "структура и размеры NIfTI"),
        ("Preprocessing", "crop, normalize, regions"),
        ("Обучение", "3D U-Net / SegResNet"),
        ("Оценка", "Dice WT, TC, ET"),
        ("Веса", "best checkpoint"),
    ]
    write_flow_svg(OUTPUT_DIR / "training_pipeline.svg", "Пайплайн обучения модели", steps)


def write_architecture_svg() -> None:
    steps = [
        ("React UI", "загрузка MRI и просмотр результата"),
        ("FastAPI", "валидация запроса и маршрутизация"),
        ("Predictor", "MONAI + PyTorch inference"),
        ("Artifacts", "seg.nii.gz и preview PNG"),
        ("Research panel", "voxel stats и Dice"),
    ]
    write_flow_svg(OUTPUT_DIR / "system_architecture.svg", "Архитектура демонстрационной системы", steps)


def write_inference_svg() -> None:
    steps = [
        ("Upload", "4 MRI-модальности"),
        ("Model", "выбор checkpoint"),
        ("Sliding window", "ROI 96x96x96"),
        ("Mask", "WT / TC / ET"),
        ("Preview", "Axial / Coronal / Sagittal"),
        ("Download", "seg.nii.gz"),
    ]
    write_flow_svg(OUTPUT_DIR / "inference_pipeline.svg", "Пайплайн инференса", steps)


def write_flow_svg(path: Path, title: str, steps: list[tuple[str, str]]) -> None:
    width = 1280
    height = 260
    card_w = 176
    gap = 28
    start_x = 38
    y = 92
    parts = [svg_header(width, height), f'<text x="38" y="44" class="title">{escape(title)}</text>']
    for index, (name, caption) in enumerate(steps):
        x = start_x + index * (card_w + gap)
        parts.append(
            f'<rect x="{x}" y="{y}" width="{card_w}" height="104" rx="8" class="card"/>'
            f'<text x="{x + 16}" y="{y + 38}" class="label">{escape(name)}</text>'
            f'<text x="{x + 16}" y="{y + 66}" class="caption">{escape(caption)}</text>'
        )
        if index < len(steps) - 1:
            ax = x + card_w + 6
            parts.append(
                f'<line x1="{ax}" y1="{y + 52}" x2="{ax + gap - 14}" y2="{y + 52}" class="arrow"/>'
                f'<path d="M {ax + gap - 14} {y + 52} l -8 -5 v 10 z" class="arrow-fill"/>'
            )
    parts.append(svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def write_model_comparison_svg(summaries: list[dict[str, object]]) -> None:
    rows = [
        item
        for item in summaries
        if item.get("mean_dice") is not None and "smoke" not in str(item.get("experiment_name", ""))
    ][:5]
    width = 900
    height = 360
    max_value = max([float(item["mean_dice"]) for item in rows] + [1.0])
    parts = [svg_header(width, height), '<text x="34" y="44" class="title">Сравнение экспериментов</text>']
    for index, item in enumerate(rows):
        y = 86 + index * 48
        value = float(item["mean_dice"])
        bar_width = int((value / max_value) * 520)
        name = short_name(str(item.get("experiment_name", "")))
        parts.append(
            f'<text x="34" y="{y + 20}" class="axis">{escape(name)}</text>'
            f'<rect x="260" y="{y}" width="520" height="26" rx="6" class="bar-bg"/>'
            f'<rect x="260" y="{y}" width="{bar_width}" height="26" rx="6" class="bar"/>'
            f'<text x="800" y="{y + 20}" class="value">{value:.3f}</text>'
        )
    parts.append(svg_footer())
    (OUTPUT_DIR / "model_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


def write_training_curve_svg(records: list[dict[str, float]]) -> None:
    points = [record for record in records if record.get("mean_dice", 0.0) > 0.0]
    width = 900
    height = 360
    left = 70
    top = 64
    chart_w = 760
    chart_h = 230
    max_epoch = max([record["epoch"] for record in points] + [1.0])
    max_dice = max([record["mean_dice"] for record in points] + [1.0])
    coords = []
    for record in points:
        x = left + record["epoch"] / max_epoch * chart_w
        y = top + chart_h - record["mean_dice"] / max_dice * chart_h
        coords.append(f"{x:.1f},{y:.1f}")
    parts = [
        svg_header(width, height),
        '<text x="34" y="44" class="title">Динамика качества baseline 3D U-Net</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" class="axis-line"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis-line"/>',
        f'<polyline points="{" ".join(coords)}" class="line"/>',
        f'<text x="{left}" y="{top + chart_h + 34}" class="axis">epoch</text>',
        f'<text x="{left - 38}" y="{top + 8}" class="axis">Dice</text>',
    ]
    parts.append(svg_footer())
    (OUTPUT_DIR / "baseline_training_curve.svg").write_text("\n".join(parts), encoding="utf-8")


def load_summaries() -> list[dict[str, object]]:
    summaries = []
    for path in sorted(METRICS_DIR.glob("*_summary.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    summaries.sort(key=lambda item: float(item.get("mean_dice") or 0.0), reverse=True)
    return summaries


def load_history(name: str) -> list[dict[str, float]]:
    path = METRICS_DIR / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file_obj:
        return [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(file_obj)
        ]


def short_name(name: str) -> str:
    if "baseline" in name:
        return "Baseline 3D U-Net"
    if "refine" in name:
        return "SegResNet v2"
    if "segresnet" in name:
        return "Transfer SegResNet"
    return name or "experiment"


def svg_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 28px Arial, sans-serif; fill: #1f343b; }}
  .label {{ font: 700 18px Arial, sans-serif; fill: #17363b; }}
  .caption {{ font: 14px Arial, sans-serif; fill: #60747a; }}
  .axis {{ font: 14px Arial, sans-serif; fill: #60747a; }}
  .value {{ font: 700 15px Arial, sans-serif; fill: #1f343b; }}
  .card {{ fill: #ffffff; stroke: #cfded9; }}
  .arrow {{ stroke: #0f766e; stroke-width: 3; stroke-linecap: round; }}
  .arrow-fill {{ fill: #0f766e; }}
  .bar-bg {{ fill: #e8f0ed; }}
  .bar {{ fill: #0f766e; }}
  .axis-line {{ stroke: #9db2ad; stroke-width: 2; }}
  .line {{ fill: none; stroke: #0f766e; stroke-width: 4; stroke-linejoin: round; }}
</style>
<rect width="100%" height="100%" fill="#f7fbfa"/>"""


def svg_footer() -> str:
    return "</svg>"


if __name__ == "__main__":
    main()
