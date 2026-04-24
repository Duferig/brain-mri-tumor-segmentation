# Brain MRI Tumor Segmentation

Исследовательский прототип для дипломной работы по автоматической сегментации опухолей головного мозга на МРТ.

## Что входит

- `PyTorch + MONAI` пайплайн для обучения baseline `3D U-Net` и improved `SwinUNETR`.
- Генератор манифестов для `BraTS`.
- Скрипты обучения, оценки и сравнения моделей.
- `FastAPI` сервис инференса с выдачей `seg.nii.gz`, preview PNG и voxel-статистики.
- `Streamlit` демо-интерфейс для загрузки 4 MRI-модальностей и просмотра результата.

## Структура

- `src/brain_mri_segmentation/ml` — данные, модели, тренировка, оценка, инференс.
- `src/brain_mri_segmentation/api` — HTTP API.
- `src/brain_mri_segmentation/ui` — Streamlit UI.
- `configs` — TOML-конфиги для обучения и инференса.
- `artifacts` — веса, метрики, предсказания и вспомогательные артефакты.
- `docs` — Colab workflow и структура дипломного текста.

## Рекомендуемое окружение

Для реальной установки нужен Python `3.10`–`3.12`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

## Быстрый старт

1. Подготовить `BraTS` и сгенерировать train/val/test manifests.
2. Обучить baseline и improved модели.
3. Запустить API.
4. Запустить Streamlit UI.

```powershell
brain-seg-manifest --dataset-root D:\data\BraTS --output-dir artifacts\manifests
brain-seg-train --config configs\train_baseline.toml
brain-seg-train --config configs\train_improved.toml
brain-seg-api --config configs\inference.toml
brain-seg-ui --config configs\inference.toml
```

## Формат данных

Для каждого случая ожидаются четыре MRI-модальности:

- `T1`
- `T1ce`
- `T2`
- `FLAIR`

Label map в формате BraTS использует метки `0`, `1`, `2`, `4`. Во время обучения label автоматически переводится в регионы `TC`, `WT`, `ET`.

## Проверки

- `python -m compileall src tests`
- `pytest`

Если зависимости для full runtime не установлены, статическая проверка синтаксиса всё равно должна проходить.

