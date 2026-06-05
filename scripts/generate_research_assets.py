from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "artifacts" / "metrics"
OUTPUT_DIR = ROOT / "output" / "research" / "assets"


@dataclass(frozen=True)
class Experiment:
    name: str
    mean_dice: float | None
    dice_wt: float | None = None
    dice_tc: float | None = None
    dice_et: float | None = None
    note: str = ""


@dataclass(frozen=True)
class TestCaseResult:
    case_id: str
    mean_dice: float
    dice_wt: float
    dice_tc: float
    dice_et: float
    seconds: float
    note: str


BENCHMARK_EXPERIMENT = Experiment(
    name="Improved SwinUNETR",
    mean_dice=0.8928,
    dice_wt=0.9245,
    dice_tc=0.9032,
    dice_et=0.8507,
    note="10 тестовых BraTS cases, improved_best.pt",
)

CHECKPOINT_EXPERIMENT = Experiment(
    name="Improved checkpoint",
    mean_dice=0.8747,
    dice_wt=0.9094,
    dice_tc=0.8838,
    dice_et=0.8310,
    note="best epoch 23, Mean HD95 8.087",
)

TEST_CASE_RESULTS = [
    TestCaseResult("BraTS-GLI-00804-000", 0.9796, 0.9841, 0.9859, 0.9688, 13.96, "самый высокий результат"),
    TestCaseResult("BraTS-GLI-00757-000", 0.8192, 0.7546, 0.8923, 0.8108, 9.06, "неравномерное качество"),
    TestCaseResult("BraTS-GLI-00285-000", 0.9353, 0.9532, 0.9257, 0.9270, 8.86, "ровные регионы"),
    TestCaseResult("BraTS-GLI-00414-000", 0.9104, 0.9690, 0.9261, 0.8361, 8.83, "сильный WT"),
    TestCaseResult("BraTS-GLI-00221-000", 0.9564, 0.9833, 0.9682, 0.9178, 8.79, "без провала ET"),
    TestCaseResult("BraTS-GLI-01459-000", 0.9252, 0.9003, 0.9297, 0.9455, 9.03, "сильный ET"),
    TestCaseResult("BraTS-GLI-00657-000", 0.9651, 0.9774, 0.9815, 0.9363, 12.44, "высокое совпадение"),
    TestCaseResult("BraTS-GLI-01428-000", 0.5149, 0.8593, 0.4532, 0.2322, 9.12, "сложный случай"),
    TestCaseResult("BraTS-GLI-00418-000", 0.9520, 0.8857, 0.9880, 0.9822, 9.39, "сильные TC и ET"),
    TestCaseResult("BraTS-GLI-00211-000", 0.9700, 0.9784, 0.9816, 0.9502, 9.05, "стабильный результат"),
]

BENCHMARK_FACTS = {
    "runs_requested": 10,
    "runs_successful": 10,
    "avg_seconds": 9.85,
    "median_seconds": 9.06,
    "peak_cuda_mb": 1406.6,
}


FALLBACK_EXPERIMENTS = [
    BENCHMARK_EXPERIMENT,
    CHECKPOINT_EXPERIMENT,
    Experiment(
        name="Baseline 3D U-Net",
        mean_dice=0.5362,
        dice_wt=0.8704,
        dice_tc=0.7382,
        dice_et=0.0,
        note="исторический локальный baseline",
    ),
]

FALLBACK_HISTORY = [
    {"epoch": 1.0, "mean_dice": 0.6113},
    {"epoch": 2.0, "mean_dice": 0.7877},
    {"epoch": 3.0, "mean_dice": 0.8193},
    {"epoch": 8.0, "mean_dice": 0.8470},
    {"epoch": 12.0, "mean_dice": 0.8627},
    {"epoch": 16.0, "mean_dice": 0.8723},
    {"epoch": 23.0, "mean_dice": 0.8747},
    {"epoch": 24.0, "mean_dice": 0.8662},
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiments = load_experiments()
    history = load_history("improved_history.csv")

    write_pipeline_svg()
    write_architecture_svg()
    write_inference_svg()
    write_model_comparison_svg(experiments)
    write_training_curve_svg(history)
    write_research_pipeline_poster()
    write_system_model_poster()
    write_experiments_poster(experiments, history)
    write_poster_index()


def write_pipeline_svg() -> None:
    steps = [
        ("BraTS", "T1 / T1ce / T2 / FLAIR"),
        ("Валидация", "NIfTI, shape, label"),
        ("Preprocess", "crop, normalize"),
        ("SwinUNETR", "4 in, 3 out"),
        ("Оценка", "Dice WT, TC, ET"),
        ("Демо", "API и preview"),
    ]
    write_flow_svg(OUTPUT_DIR / "training_pipeline.svg", "Пайплайн подготовки и проверки модели", steps)


def write_architecture_svg() -> None:
    steps = [
        ("React UI", "загрузка MRI и просмотр результата"),
        ("FastAPI", "валидация и маршруты"),
        ("Preprocess", "4 канала, ROI 96"),
        ("SwinUNETR", "MONAI + PyTorch"),
        ("Artifacts", "seg.nii.gz и preview PNG"),
    ]
    write_flow_svg(OUTPUT_DIR / "system_architecture.svg", "Архитектура демо-системы", steps)


def write_inference_svg() -> None:
    steps = [
        ("Upload", "4 MRI-модальности"),
        ("Model", "improved_best.pt"),
        ("Sliding window", "ROI 96x96x96"),
        ("Mask", "WT / TC / ET"),
        ("Preview", "3 ортогональные проекции"),
        ("Download", "seg.nii.gz"),
    ]
    write_flow_svg(OUTPUT_DIR / "inference_pipeline.svg", "Пайплайн инференса", steps)


def write_flow_svg(path: Path, title: str, steps: list[tuple[str, str]]) -> None:
    width = 1280
    height = 300
    card_w = 174
    gap = 32
    start_x = 34
    y = 116
    parts = [svg_header(width, height), text(38, 50, title, "title")]
    parts.append(text(38, 78, "Лаконичная версия для вставки в пояснительную записку", "caption"))

    for index, (name, caption) in enumerate(steps):
        x = start_x + index * (card_w + gap)
        parts.append(rect(x, y, card_w, 118, 10, "card"))
        parts.append(text(x + 16, y + 35, f"{index + 1:02d}", "number"))
        parts.append(text(x + 16, y + 66, name, "label"))
        parts.extend(wrapped_text(x + 16, y + 91, caption, 19, "caption", 18, 2))
        if index < len(steps) - 1:
            ax = x + card_w
            arrow_y = y + 59
            parts.append(
                f'<line x1="{ax}" y1="{arrow_y}" x2="{ax + gap - 8}" '
                'y2="{arrow_y}" class="arrow"/>'.format(arrow_y=arrow_y)
            )
            parts.append(
                f'<path d="M {ax + gap - 8} {arrow_y} l -9 -6 v 12 z" class="arrow-fill"/>'
            )
    parts.append(svg_footer())
    path.write_text("\n".join(parts), encoding="utf-8")


def write_model_comparison_svg(experiments: list[Experiment]) -> None:
    rows = [item for item in experiments if item.mean_dice is not None][:5]
    width = 980
    height = 380
    max_value = max([item.mean_dice or 0.0 for item in rows] + [1.0])
    parts = [svg_header(width, height), text(34, 48, "Сравнение экспериментов", "title")]
    parts.append(text(34, 76, "Основная метрика: средний Dice по регионам WT, TC, ET", "caption"))
    for index, item in enumerate(rows):
        y = 116 + index * 64
        value = item.mean_dice or 0.0
        bar_width = int((value / max_value) * 560)
        parts.append(text(34, y + 21, item.name, "axis"))
        parts.append(rect(296, y, 560, 30, 7, "bar-bg"))
        parts.append(rect(296, y, bar_width, 30, 7, "bar"))
        parts.append(text(878, y + 22, f"{value:.3f}", "value"))
        if item.note:
            parts.append(text(296, y + 51, item.note, "caption"))
    parts.append(svg_footer())
    (OUTPUT_DIR / "model_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


def write_training_curve_svg(records: list[dict[str, float]]) -> None:
    points = [record for record in records if record.get("mean_dice", 0.0) > 0.0]
    width = 980
    height = 380
    left = 82
    top = 78
    chart_w = 790
    chart_h = 232
    max_epoch = max([record["epoch"] for record in points] + [1.0])
    max_dice = max([record["mean_dice"] for record in points] + [1.0])
    coords = []
    for record in points:
        x = left + record["epoch"] / max_epoch * chart_w
        y = top + chart_h - record["mean_dice"] / max_dice * chart_h
        coords.append(f"{x:.1f},{y:.1f}")

    parts = [
        svg_header(width, height),
        text(34, 48, "Динамика качества Improved SwinUNETR", "title"),
        text(34, 76, "Mean Dice на best checkpoint: 0.8747, benchmark на 10 cases: 0.8928", "caption"),
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" '
        f'y2="{top + chart_h}" class="axis-line"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" class="axis-line"/>',
        f'<polyline points="{" ".join(coords)}" class="line"/>',
        text(left, top + chart_h + 38, "epoch", "axis"),
        text(left - 44, top + 8, "Dice", "axis"),
    ]
    for record in points[-3:]:
        x = left + record["epoch"] / max_epoch * chart_w
        y = top + chart_h - record["mean_dice"] / max_dice * chart_h
        parts.append(circle(x, y, 5, "point"))
    parts.append(svg_footer())
    (OUTPUT_DIR / "baseline_training_curve.svg").write_text("\n".join(parts), encoding="utf-8")


def write_research_pipeline_poster() -> None:
    width = 1684
    height = 1191
    parts = [poster_header(width, height)]
    add_poster_title(
        parts,
        "Постановка задачи и данные для сегментации МРТ",
        "Что принимает система, какие области опухоли выделяет и какие артефакты возвращает пользователю",
        "Плакат 1 / постановка задачи",
    )

    parts.append(panel(72, 246, 520, 390, "Входной MRI-набор", "Один case BraTS, четыре согласованные модальности"))
    parts.extend(
        bullet_list(
            104,
            352,
            [
                "T1: анатомическая структура ткани",
                "T1ce: контрастное усиление опухоли",
                "T2: отек и жидкостные изменения",
                "FLAIR: подавление ликвора и зона поражения",
            ],
            max_chars=42,
        )
    )
    for index, modality in enumerate(["T1", "T1ce", "T2", "FLAIR"]):
        parts.append(modality_card(108 + index * 126, 520, modality, index))

    parts.append(panel(632, 246, 980, 390, "Целевые области", "Региональное представление BraTS для медицинской сегментации"))
    parts.append(region_chip(690, 344, "WT", "Whole tumor", "wt"))
    parts.append(region_chip(846, 344, "TC", "Tumor core", "tc"))
    parts.append(region_chip(1002, 344, "ET", "Enhancing tumor", "et"))
    parts.append(mask_preview(690, 448, 336, 110))
    parts.extend(
        bullet_list(
            1098,
            454,
            [
                "модель предсказывает 3 региональные маски",
                "после инференса регионы переводятся в label map",
                "ET поддерживается как label 3 или label 4",
            ],
            max_chars=44,
        )
    )

    parts.append(panel(72, 676, 886, 340, "Контур обработки", "От загрузки MRI до результата в интерфейсе"))
    pipeline = [
        ("01", "Загрузка", "4 томограммы"),
        ("02", "Проверка", "размеры, файлы"),
        ("03", "Нормализация", "область мозга"),
        ("04", "Сегментация", "нейросеть"),
        ("05", "Области", "WT TC ET"),
        ("06", "Вывод", "маска и снимки"),
    ]
    for index, (number, label, caption) in enumerate(pipeline):
        x = 112 + index * 132
        y = 792 if index % 2 == 0 else 834
        parts.append(process_node(x, y, number, label, caption))

    parts.append(panel(956, 676, 656, 340, "Выходные артефакты", "Файлы и показатели, которые сохраняет прототип"))
    parts.extend(
        numbered_list(
            1004,
            782,
            [
                "файл с итоговой маской опухоли",
                "изображения в трех анатомических плоскостях",
                "voxel statistics по регионам WT, TC и ET",
                "Dice-метрики при наличии эталонной маски",
            ],
            max_chars=48,
        )
    )
    parts.append(callout(1098, 954, 410, 54, "Назначение", "исследовательский прототип для демонстрации ML-pipeline"))

    parts.append(poster_footer("BraTS MRI / регионы TC-WT-ET / маска, изображения и метрики"))
    parts.append(svg_footer())
    (OUTPUT_DIR / "poster_01_research_pipeline.svg").write_text("\n".join(parts), encoding="utf-8")


def write_system_model_poster() -> None:
    width = 1684
    height = 1191
    parts = [poster_header(width, height)]
    add_poster_title(
        parts,
        "Архитектура системы и модели SwinUNETR",
        "Компоненты программного прототипа, поток данных и параметры основной демонстрационной модели",
        "Плакат 2 / архитектура системы",
    )

    parts.append(panel(72, 246, 1540, 326, "Модульная архитектура", "Поток показан справа налево: от действия пользователя к готовым результатам"))
    layers = [
        ("Результаты", "маска, снимки, метрики", "artifact"),
        ("Постобработка", "области WT, TC, ET", "ui"),
        ("Нейросеть", "SwinUNETR, 3D объем", "model"),
        ("Подготовка", "нормализация и ROI", "api"),
        ("Проверка", "модальности и размеры", "ui"),
        ("Сервер", "прием и управление", "api"),
        ("Интерфейс", "загрузка и просмотр", "ui"),
        ("Пользователь", "выбирает набор МРТ", "artifact"),
    ]
    module_w = 150
    gap = 38
    start_x = 96
    for index, (title, caption, tone) in enumerate(layers):
        x = start_x + index * (module_w + gap)
        parts.append(flow_module(x, 370, module_w, title, caption, tone))
        if index < len(layers) - 1:
            parts.append(arrow(x + module_w + gap - 4, 440, x + module_w + 8, 440, "arrow"))

    parts.append(panel(72, 624, 760, 392, "Основная модель", "MONAI SwinUNETR для объемной сегментации"))
    parts.extend(swinunetr_diagram(128, 738))
    parts.extend(
        bullet_list(
            592,
            730,
            [
                "4 входных канала: T1, T1ce, T2, FLAIR",
                "3 выходных канала: TC, WT и ET",
                "ROI инференса 96x96x96 voxel",
                "около 15.7 млн обучаемых параметров",
            ],
            max_chars=28,
        )
    )

    parts.append(panel(872, 624, 740, 392, "Модули программной системы", "Функциональные части без привязки к именам файлов"))
    parts.append(model_stack(930, 734))
    parts.extend(
        bullet_list(
            1198,
            738,
            [
                "модуль данных разделяет обучение и тестирование",
                "модуль модели хранит обученные веса",
                "модуль результатов сохраняет каждый запуск отдельно",
                "интерфейс показывает готовые артефакты пользователю",
            ],
            max_chars=40,
        )
    )
    parts.append(callout(1010, 940, 492, 72, "Идея", "каждый модуль отвечает за один понятный этап обработки"))

    parts.append(poster_footer("Модульная система: интерфейс, сервер, подготовка данных, модель, результаты"))
    parts.append(svg_footer())
    (OUTPUT_DIR / "poster_02_system_model_architecture.svg").write_text("\n".join(parts), encoding="utf-8")


def write_experiments_poster(experiments: list[Experiment], history: list[dict[str, float]]) -> None:
    width = 1684
    height = 1191
    parts = [poster_header(width, height)]
    add_poster_title(
        parts,
        "Тестирование и результаты модели",
        "Проверка Improved SwinUNETR на 10 тестовых случаях: качество сегментации, время работы и разброс результатов",
        "Плакат 3 / тестирование",
    )

    metrics = [
        ("Mean Dice", BENCHMARK_EXPERIMENT.mean_dice, "среднее WT/TC/ET", "metric-main"),
        ("Dice WT", BENCHMARK_EXPERIMENT.dice_wt, "whole tumor", "metric-wt"),
        ("Dice TC", BENCHMARK_EXPERIMENT.dice_tc, "tumor core", "metric-tc"),
        ("Dice ET", BENCHMARK_EXPERIMENT.dice_et, "enhancing tumor", "metric-et"),
    ]
    for index, (title, value, subtitle, klass) in enumerate(metrics):
        parts.append(metric_card(72 + index * 386, 248, 348, 158, title, value or 0.0, subtitle, klass))

    parts.append(panel(72, 446, 520, 314, "Методика испытаний", "Тестовый контур повторяет пользовательский сценарий"))
    parts.extend(
        bullet_list(
            108,
            548,
            [
                "10 случаев из отдельной тестовой выборки",
                "запуск через основную SwinUNETR-модель",
                f"успешных прогонов: {BENCHMARK_FACTS['runs_successful']} из {BENCHMARK_FACTS['runs_requested']}",
                f"среднее время: {BENCHMARK_FACTS['avg_seconds']:.2f} с, медиана: {BENCHMARK_FACTS['median_seconds']:.2f} с",
            ],
            max_chars=43,
        )
    )

    parts.append(panel(632, 446, 980, 314, "Качество по 10 случаям", "Mean Dice для каждого case, включая сложный пограничный пример"))
    parts.extend(case_distribution_chart(684, 552, 846, 150, TEST_CASE_RESULTS))
    parts.append(text(684, 744, "лучший case 0.980; сложный case 0.515; среднее 0.893", "small"))

    parts.append(panel(72, 800, 742, 260, "Динамика обучения", "лучшая эпоха обучения - 23"))
    parts.extend(training_curve(132, 888, 592, 80, history))
    parts.append(text(132, 1022, "лучшая эпоха 23: Mean Dice 0.8747, Mean HD95 8.087", "small"))

    parts.append(panel(850, 800, 762, 260, "Интерпретация результата", "Что подтверждает тестирование"))
    parts.extend(
        numbered_list(
            902,
            910,
            [
                "WT, TC и ET устойчивы на большинстве тестовых cases",
                "Dice ET = 0.851; нулевого провала нет",
                "качество зависит от case, прототип исследовательский",
            ],
            max_chars=50,
        )
    )

    parts.append(poster_footer("10 тестовых случаев: Mean Dice 0.8928 / WT 0.9245 / TC 0.9032 / ET 0.8507"))
    parts.append(svg_footer())
    (OUTPUT_DIR / "poster_03_experiments_results.svg").write_text("\n".join(parts), encoding="utf-8")


def write_poster_index() -> None:
    posters = [
        "poster_01_research_pipeline.svg",
        "poster_02_system_model_architecture.svg",
        "poster_03_experiments_results.svg",
    ]
    cards = "\n".join(
        f'<article><h2>{escape(path)}</h2><img src="{escape(path)}" alt="{escape(path)}"></article>'
        for path in posters
    )
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research Posters Preview</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #eef3f1; color: #15211f; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 32px; }}
    h1 {{ margin: 0 0 24px; font-size: 28px; }}
    article {{ margin: 0 0 40px; }}
    h2 {{ font-size: 16px; font-weight: 650; margin: 0 0 10px; }}
    img {{ width: 100%; display: block; background: white; box-shadow: 0 18px 60px rgba(20, 43, 39, .18); }}
  </style>
</head>
<body>
  <main>
    <h1>Research project posters</h1>
    {cards}
  </main>
</body>
</html>"""
    (OUTPUT_DIR / "poster_index.html").write_text(html, encoding="utf-8")


def load_experiments() -> list[Experiment]:
    experiments = [BENCHMARK_EXPERIMENT, CHECKPOINT_EXPERIMENT]
    seen = {item.name for item in experiments}
    for path in sorted(METRICS_DIR.glob("*_summary.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = short_name(str(raw.get("experiment_name", path.stem)))
        if "smoke" in name.lower() or name in seen:
            continue
        seen.add(name)
        experiments.append(
            Experiment(
                name=name,
                mean_dice=as_float(raw.get("mean_dice")),
                dice_wt=as_float(raw.get("dice_wt")),
                dice_tc=as_float(raw.get("dice_tc")),
                dice_et=as_float(raw.get("dice_et")),
            )
        )
    if len(experiments) == 2:
        experiments.extend(item for item in FALLBACK_EXPERIMENTS if item.name not in seen)
    return sorted(experiments, key=lambda item: item.mean_dice or 0.0, reverse=True)


def load_history(name: str) -> list[dict[str, float]]:
    path = METRICS_DIR / name
    if not path.exists():
        return FALLBACK_HISTORY
    with path.open(encoding="utf-8", newline="") as file_obj:
        records = [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(file_obj)
        ]
    return records or FALLBACK_HISTORY


def as_float(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def short_name(name: str) -> str:
    lower = name.lower()
    if "baseline" in lower:
        return "Baseline 3D U-Net"
    if "refine" in lower:
        return "SegResNet v2"
    if "segresnet" in lower:
        return "Transfer SegResNet"
    if "swin" in lower:
        return "Improved SwinUNETR"
    return name or "experiment"


def best_experiment(experiments: list[Experiment]) -> Experiment:
    with_regions = [item for item in experiments if item.dice_wt is not None]
    return with_regions[0] if with_regions else FALLBACK_EXPERIMENTS[0]


def poster_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <linearGradient id="poster-bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#f7faf9"/>
    <stop offset="0.58" stop-color="#edf4f1"/>
    <stop offset="1" stop-color="#f6f0ea"/>
  </linearGradient>
  <pattern id="grid" width="38" height="38" patternUnits="userSpaceOnUse">
    <path d="M 38 0 L 0 0 0 38" fill="none" stroke="#d9e5e1" stroke-width="1" opacity="0.38"/>
  </pattern>
</defs>
{poster_styles()}
<rect width="100%" height="100%" fill="url(#poster-bg)"/>
<rect width="100%" height="100%" fill="url(#grid)" opacity="0.42"/>"""


def poster_styles() -> str:
    return """<style>
  text { font-family: 'Segoe UI', Arial, sans-serif; letter-spacing: 0; }
  .poster-kicker { font-size: 19px; font-weight: 700; fill: #a4552a; }
  .poster-title { font-size: 40px; font-weight: 800; fill: #15211f; }
  .poster-subtitle { font-size: 23px; font-weight: 500; fill: #4c625e; }
  .poster-footer { font-size: 18px; font-weight: 600; fill: #58706b; }
  .panel { fill: rgba(255,255,255,.78); stroke: #c6d8d3; stroke-width: 1.4; }
  .panel-title { font-size: 24px; font-weight: 760; fill: #182823; }
  .panel-subtitle { font-size: 16px; font-weight: 600; fill: #68807b; }
  .body { font-size: 18px; font-weight: 500; fill: #263935; }
  .small { font-size: 14px; font-weight: 550; fill: #5c746e; }
  .section-kicker { font-size: 18px; font-weight: 760; fill: #9b4d2f; }
  .bullet-dot { fill: #0f766e; }
  .node { fill: #fdfefe; stroke: #9ebbb4; stroke-width: 1.3; }
  .node-number { font-size: 18px; font-weight: 800; fill: #0f766e; }
  .node-title { font-size: 17px; font-weight: 780; fill: #172824; }
  .node-caption { font-size: 12px; font-weight: 550; fill: #5a716b; }
  .chip { fill: #ffffff; stroke-width: 1.2; }
  .chip-main { font-size: 20px; font-weight: 820; fill: #15211f; }
  .chip-sub { font-size: 12px; font-weight: 680; fill: #667d77; }
  .arrow, .thin-arrow { fill: none; stroke: #0f766e; stroke-linecap: round; stroke-linejoin: round; }
  .arrow { stroke-width: 3.2; }
  .thin-arrow { stroke-width: 2.2; opacity: .88; }
  .muted-box { fill: #f4f8f6; stroke: #bfd4ce; stroke-width: 1.2; }
  .ui { fill: #e8f4f1; stroke: #0f766e; }
  .api { fill: #edf2fb; stroke: #315f96; }
  .model { fill: #fff2e6; stroke: #b26a2e; }
  .artifact { fill: #f8eef0; stroke: #9a3e47; }
  .bar-bg-poster { fill: #dfe9e6; }
  .bar-poster { fill: #0f766e; }
  .bar-alt { fill: #315f96; }
  .bar-warn { fill: #b26a2e; }
  .callout { fill: #15211f; }
  .callout-title { font-size: 13px; font-weight: 800; fill: #f6c177; }
  .callout-text { font-size: 15px; font-weight: 620; fill: #f7fbfa; }
  .curve { fill: none; stroke: #0f766e; stroke-width: 4; stroke-linejoin: round; }
  .axis-line-poster { stroke: #99b4ad; stroke-width: 1.5; }
  .metric-card { fill: rgba(255,255,255,.86); stroke: #c6d8d3; stroke-width: 1.4; }
  .metric-main { fill: #0f766e; }
  .metric-wt { fill: #2f7d62; }
  .metric-tc { fill: #315f96; }
  .metric-et { fill: #b26a2e; }
  .metric-title { font-size: 18px; font-weight: 760; fill: #38524d; }
  .metric-value { font-size: 43px; font-weight: 840; fill: #14231f; }
  .metric-subtitle { font-size: 13px; font-weight: 660; fill: #667d77; }
  .case-label { font-size: 12px; font-weight: 680; fill: #4f6761; }
  .case-value { font-size: 13px; font-weight: 760; fill: #14231f; }
  .case-bar-bg { fill: #dfe9e6; }
  .case-bar-good { fill: #0f766e; }
  .case-bar-mid { fill: #315f96; }
  .case-bar-low { fill: #b26a2e; }
</style>"""


def svg_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: 'Segoe UI', Arial, sans-serif; letter-spacing: 0; }}
  .title {{ font-size: 29px; font-weight: 760; fill: #1b302c; }}
  .label {{ font-size: 18px; font-weight: 740; fill: #17363b; }}
  .caption {{ font-size: 14px; font-weight: 500; fill: #60747a; }}
  .axis {{ font-size: 15px; font-weight: 560; fill: #4d6560; }}
  .value {{ font-size: 16px; font-weight: 780; fill: #1f343b; }}
  .number {{ font-size: 15px; font-weight: 820; fill: #0f766e; }}
  .card {{ fill: #ffffff; stroke: #c9dbd6; stroke-width: 1.2; }}
  .arrow {{ stroke: #0f766e; stroke-width: 3; stroke-linecap: round; }}
  .arrow-fill {{ fill: #0f766e; }}
  .bar-bg {{ fill: #e3ede9; }}
  .bar {{ fill: #0f766e; }}
  .axis-line {{ stroke: #9db2ad; stroke-width: 2; }}
  .line {{ fill: none; stroke: #0f766e; stroke-width: 4; stroke-linejoin: round; }}
  .point {{ fill: #b26a2e; stroke: #ffffff; stroke-width: 2; }}
</style>
<rect width="100%" height="100%" fill="#f7fbfa"/>"""


def add_poster_title(parts: list[str], title: str, subtitle: str, kicker: str) -> None:
    parts.append(text(86, 104, kicker, "poster-kicker"))
    title_lines = wrap_text(title, 86)[:2]
    for index, line in enumerate(title_lines):
        parts.append(text(86, 168 + index * 62, line, "poster-title"))
    subtitle_y = 222 + max(0, len(title_lines) - 1) * 62
    parts.extend(wrapped_text(88, subtitle_y, subtitle, 120, "poster-subtitle", 32, 1))


def poster_footer(content: str) -> str:
    return text(86, 1114, content, "poster-footer") + text(1416, 1114, "Brain MRI Segmentation", "poster-footer")


def panel(x: int, y: int, width: int, height: int, title_value: str, subtitle: str) -> str:
    return "\n".join(
        [
            rect(x, y, width, height, 16, "panel"),
            text(x + 30, y + 48, title_value, "panel-title"),
            text(x + 30, y + 76, subtitle, "panel-subtitle"),
        ]
    )


def bullet_list(x: int, y: int, items: list[str], max_chars: int) -> list[str]:
    parts: list[str] = []
    cursor = y
    for item in items:
        lines = wrap_text(item, max_chars)
        parts.append(circle(x, cursor - 8, 5, "bullet-dot"))
        for line in lines:
            parts.append(text(x + 22, cursor, line, "body"))
            cursor += 24
        cursor += 10
    return parts


def numbered_list(x: int, y: int, items: list[str], max_chars: int) -> list[str]:
    parts: list[str] = []
    cursor = y
    for index, item in enumerate(items, start=1):
        parts.append(rect(x, cursor - 20, 34, 34, 9, "muted-box"))
        parts.append(text(x + 10, cursor + 4, str(index), "node-number"))
        lines = wrap_text(item, max_chars)
        for line in lines:
            parts.append(text(x + 52, cursor, line, "body"))
            cursor += 24
        cursor += 16
    return parts


def process_node(x: int, y: int, number: str, title_value: str, caption: str) -> str:
    return "\n".join(
        [
            rect(x, y, 128, 88, 14, "node"),
            text(x + 16, y + 28, number, "node-number"),
            text(x + 16, y + 54, title_value, "node-title"),
            text(x + 16, y + 76, caption, "node-caption"),
        ]
    )


def region_chip(x: int, y: int, label: str, caption: str, tone: str) -> str:
    stroke = {"wt": "#0f766e", "tc": "#315f96", "et": "#b26a2e"}[tone]
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="112" height="56" rx="13" class="chip" stroke="{stroke}"/>',
            text(x + 16, y + 25, label, "chip-main"),
            text(x + 16, y + 44, caption, "chip-sub"),
        ]
    )


def modality_card(x: int, y: int, label: str, index: int) -> str:
    fills = ["#d9ebe7", "#dfe8f7", "#f5e8dc", "#f2dfe3"]
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="122" height="122" rx="18" fill="{fills[index]}" '
            'stroke="#a8bfba" stroke-width="1.2"/>',
            f'<path d="M {x + 26} {y + 86} C {x + 48} {y + 26}, {x + 82} {y + 30}, '
            f'{x + 96} {y + 86}" fill="none" stroke="#15211f" stroke-width="3" opacity="0.62"/>',
            circle(x + 62, y + 60, 24, "bullet-dot"),
            text(x + 38, y + 108, label, "node-title"),
        ]
    )


def mask_preview(x: int, y: int, width: int, height: int) -> str:
    return "\n".join(
        [
            rect(x, y, width, height, 18, "muted-box"),
            f'<ellipse cx="{x + 74}" cy="{y + 46}" rx="46" ry="34" fill="#dfe8f7" stroke="#315f96" stroke-width="2"/>',
            f'<path d="M {x + 50} {y + 42} C {x + 74} {y + 18}, {x + 120} {y + 32}, {x + 106} {y + 66} C {x + 90} {y + 90}, {x + 50} {y + 74}, {x + 50} {y + 42} Z" fill="#0f766e" opacity="0.72"/>',
            f'<circle cx="{x + 92}" cy="{y + 52}" r="18" fill="#b26a2e" opacity="0.86"/>',
            text(x + 150, y + 39, "mask", "node-title"),
            text(x + 150, y + 66, "WT / TC / ET", "node-caption"),
        ]
    )


def metric_card(
    x: int,
    y: int,
    width: int,
    height: int,
    title_value: str,
    value: float,
    subtitle: str,
    tone: str,
) -> str:
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" class="metric-card"/>',
            f'<rect x="{x}" y="{y}" width="12" height="{height}" rx="6" class="{tone}"/>',
            text(x + 34, y + 44, title_value, "metric-title"),
            text(x + 34, y + 98, f"{value:.3f}", "metric-value"),
            text(x + 34, y + 132, subtitle, "metric-subtitle"),
        ]
    )


def case_distribution_chart(x: int, y: int, width: int, height: int, cases: list[TestCaseResult]) -> list[str]:
    parts: list[str] = []
    row_h = 13
    gap = 4
    label_w = 158
    value_w = 48
    bar_w = width - label_w - value_w - 28
    for index, case in enumerate(cases):
        yy = y + index * (row_h + gap)
        short_case = case.case_id.replace("BraTS-GLI-", "")
        klass = "case-bar-good" if case.mean_dice >= 0.9 else "case-bar-mid" if case.mean_dice >= 0.75 else "case-bar-low"
        parts.append(text(x, yy + 11, short_case, "case-label"))
        parts.append(rect(x + label_w, yy, bar_w, row_h, 5, "case-bar-bg"))
        parts.append(rect(x + label_w, yy, bar_w * case.mean_dice, row_h, 5, klass))
        parts.append(text(x + label_w + bar_w + 18, yy + 11, f"{case.mean_dice:.3f}", "case-value"))
    return parts


def system_layer(x: int, y: int, title_value: str, caption: str, tone: str) -> str:
    parts = [f'<rect x="{x}" y="{y}" width="266" height="150" rx="18" class="{tone}" stroke-width="1.6"/>']
    parts.append(text(x + 24, y + 48, title_value, "panel-title"))
    parts.extend(wrapped_text(x + 24, y + 84, caption, 29, "small", 22, 3))
    return "\n".join(parts)


def flow_module(x: int, y: int, width: int, title_value: str, caption: str, tone: str) -> str:
    parts = [f'<rect x="{x}" y="{y}" width="{width}" height="138" rx="16" class="{tone}" stroke-width="1.4"/>']
    parts.extend(wrapped_text(x + 18, y + 42, title_value, 14, "node-title", 21, 2))
    parts.extend(wrapped_text(x + 18, y + 82, caption, 17, "node-caption", 18, 3))
    return "\n".join(parts)


def unet_diagram(x: int, y: int) -> list[str]:
    parts: list[str] = []
    heights = [132, 106, 82, 62]
    for index, h in enumerate(heights):
        bx = x + index * 54
        by = y + (132 - h) / 2
        parts.append(f'<rect x="{bx}" y="{by}" width="40" height="{h}" rx="8" class="model" stroke-width="1.3"/>')
    for index, h in enumerate(reversed(heights[:-1])):
        bx = x + 246 + index * 54
        by = y + (132 - h) / 2
        parts.append(f'<rect x="{bx}" y="{by}" width="40" height="{h}" rx="8" class="ui" stroke-width="1.3"/>')
    parts.append(arrow(x + 40, y + 66, x + 240, y + 66, "thin-arrow"))
    parts.append(f'<path d="M {x + 20} {y + 16} C {x + 132} {y - 24}, {x + 286} {y - 22}, {x + 344} {y + 18}" class="thin-arrow"/>')
    parts.append(f'<path d="M {x + 74} {y + 38} C {x + 170} {y + 12}, {x + 270} {y + 12}, {x + 292} {y + 42}" class="thin-arrow"/>')
    parts.append(text(x, y + 174, "encoder", "small"))
    parts.append(text(x + 282, y + 174, "decoder / skip", "small"))
    return parts


def swinunetr_diagram(x: int, y: int) -> list[str]:
    parts: list[str] = []
    stages = [
        ("Patch", "4ch", 92, "api"),
        ("Swin 1", "24", 116, "ui"),
        ("Swin 2", "48", 138, "ui"),
        ("Swin 3", "96", 160, "model"),
        ("Swin 4", "192", 182, "model"),
    ]
    for index, (label, channels, h, tone) in enumerate(stages):
        bx = x + index * 50
        by = y + (182 - h) / 2
        parts.append(f'<rect x="{bx}" y="{by}" width="36" height="{h}" rx="9" class="{tone}" stroke-width="1.3"/>')
        parts.append(text(bx + 5, by + 27, label, "node-caption"))
        parts.append(text(bx + 8, by + h - 14, channels, "node-number"))
    for index, h in enumerate([158, 134, 110, 86]):
        bx = x + 282 + index * 36
        by = y + (182 - h) / 2
        parts.append(f'<rect x="{bx}" y="{by}" width="30" height="{h}" rx="8" class="ui" stroke-width="1.3"/>')
    parts.append(arrow(x + 238, y + 91, x + 272, y + 91, "thin-arrow"))
    parts.append(f'<path d="M {x + 32} {y + 26} C {x + 142} {y - 34}, {x + 344} {y - 30}, {x + 424} {y + 20}" class="thin-arrow"/>')
    parts.append(f'<path d="M {x + 86} {y + 54} C {x + 188} {y + 12}, {x + 320} {y + 14}, {x + 376} {y + 58}" class="thin-arrow"/>')
    parts.append(text(x, y + 220, "Swin Transformer encoder", "small"))
    parts.append(text(x + 282, y + 220, "U-Net decoder + skip", "small"))
    return parts


def model_stack(x: int, y: int) -> str:
    items = [
        ("Данные", "обучение / проверка / тест"),
        ("Модель", "основные обученные веса"),
        ("Результаты", "маска и изображения"),
    ]
    parts: list[str] = []
    for index, (title_value, caption) in enumerate(items):
        yy = y + index * 76
        parts.append(rect(x + index * 18, yy, 214, 56, 12, "muted-box"))
        parts.append(text(x + 18 + index * 18, yy + 24, title_value, "node-title"))
        parts.append(text(x + 18 + index * 18, yy + 44, caption, "node-caption"))
    return "\n".join(parts)


def results_chart(x: int, y: int, width: int, height: int, experiments: list[Experiment]) -> list[str]:
    rows = [item for item in experiments if item.mean_dice is not None][:3]
    max_value = max([item.mean_dice or 0.0 for item in rows] + [1.0])
    parts: list[str] = []
    for index, item in enumerate(rows):
        yy = y + index * 74
        value = item.mean_dice or 0.0
        bar_width = int((value / max_value) * width)
        klass = "bar-poster" if index == 0 else "bar-alt"
        parts.append(text(x, yy - 12, item.name, "body"))
        parts.append(rect(x, yy, width, 32, 8, "bar-bg-poster"))
        parts.append(rect(x, yy, bar_width, 32, 8, klass))
        parts.append(text(x + width + 22, yy + 24, f"{value:.3f}", "body"))
    return parts


def region_bars(x: int, y: int, experiment: Experiment) -> list[str]:
    rows = [
        ("WT", "Whole tumor", experiment.dice_wt, "bar-poster"),
        ("TC", "Tumor core", experiment.dice_tc, "bar-alt"),
        ("ET", "Enhancing tumor", experiment.dice_et, "bar-warn"),
    ]
    parts: list[str] = []
    for index, (label, caption, value, klass) in enumerate(rows):
        yy = y + index * 72
        safe_value = value if value is not None else 0.0
        parts.append(text(x, yy + 24, label, "panel-title"))
        parts.append(text(x + 62, yy + 22, caption, "small"))
        parts.append(rect(x + 248, yy, 326, 30, 8, "bar-bg-poster"))
        parts.append(rect(x + 248, yy, int(326 * safe_value), 30, 8, klass))
        parts.append(text(x + 596, yy + 23, f"{safe_value:.3f}", "body"))
    return parts


def training_curve(x: int, y: int, width: int, height: int, records: list[dict[str, float]]) -> list[str]:
    points = [record for record in records if record.get("mean_dice", 0.0) > 0.0]
    max_epoch = max([record["epoch"] for record in points] + [1.0])
    max_dice = max([record["mean_dice"] for record in points] + [1.0])
    coords = []
    for record in points:
        px = x + record["epoch"] / max_epoch * width
        py = y + height - record["mean_dice"] / max_dice * height
        coords.append(f"{px:.1f},{py:.1f}")
    return [
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" class="axis-line-poster"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" class="axis-line-poster"/>',
        f'<polyline points="{" ".join(coords)}" class="curve"/>',
        text(x, y + height + 34, "epoch", "small"),
        text(x + width - 78, y - 12, "mean Dice", "small"),
    ]


def callout(x: int, y: int, width: int, height: int, title_value: str, body: str) -> str:
    title_y = y + 24 if height <= 60 else y + 28
    body_y = y + 44 if height <= 60 else y + 54
    parts = [rect(x, y, width, height, 14, "callout"), text(x + 22, title_y, title_value, "callout-title")]
    parts.extend(wrapped_text(x + 22, body_y, body, max(width // 9, 26), "callout-text", 20, 2))
    return "\n".join(parts)


def arrow(x1: float, y1: float, x2: float, y2: float, klass: str) -> str:
    return (
        f'<path d="M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}" class="{klass}"/>'
        f'<path d="M {x2:.1f} {y2:.1f} l -10 -6 v 12 z" fill="#0f766e"/>'
    )


def curved_arrow(x1: float, y1: float, x2: float, y2: float, klass: str) -> str:
    cx = (x1 + x2) / 2
    return (
        f'<path d="M {x1:.1f} {y1:.1f} C {cx:.1f} {y1:.1f}, {cx:.1f} {y2:.1f}, '
        f'{x2:.1f} {y2:.1f}" class="{klass}"/>'
    )


def wrapped_text(
    x: float,
    y: float,
    value: str,
    max_chars: int,
    klass: str,
    line_height: int,
    max_lines: int,
) -> list[str]:
    lines = wrap_text(value, max_chars)[:max_lines]
    return [text(x, y + index * line_height, line, klass) for index, line in enumerate(lines)]


def wrap_text(value: str, max_chars: int) -> list[str]:
    return wrap(value, width=max_chars, break_long_words=False, break_on_hyphens=False) or [value]


def rect(x: float, y: float, width: float, height: float, radius: float, klass: str) -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" class="{klass}"/>'


def circle(x: float, y: float, radius: float, klass: str) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{radius}" class="{klass}"/>'


def text(x: float, y: float, value: str, klass: str) -> str:
    return f'<text x="{x}" y="{y}" class="{klass}">{escape(value)}</text>'


def svg_footer() -> str:
    return "</svg>"


if __name__ == "__main__":
    main()
