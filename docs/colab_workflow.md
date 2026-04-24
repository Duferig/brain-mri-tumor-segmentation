# Colab Workflow

## Цель

Использовать Google Colab только для обучения и оценки, а локально держать API и Streamlit-демо.

## Шаги

1. Смонтировать Google Drive.
2. Клонировать проект или загрузить архив.
3. Установить зависимости проекта в Python 3.10/3.11 runtime.
4. Сгенерировать manifests для `BraTS`.
5. Запустить обучение baseline и improved конфигов.
6. Скопировать лучшие веса и `summary.json` обратно в `artifacts/models` и `artifacts/metrics`.

## Минимальный сценарий

```python
!pip install -e .[dev]
!brain-seg-manifest --dataset-root /content/BraTS --output-dir artifacts/manifests
!brain-seg-train --config configs/train_baseline.toml
!brain-seg-train --config configs/train_improved.toml
!brain-seg-eval --config configs/train_baseline.toml
!brain-seg-eval --config configs/train_improved.toml
```

## Практические замечания

- Для `SwinUNETR` обычно требуется меньше `batch_size`, чем для baseline `3D U-Net`.
- Если Colab GPU ограничен по памяти, сначала уменьшать `batch_size`, затем `sw_batch_size`.
- ROI `96x96x96` сохраняется как базовый размер, если нет критической нехватки памяти.

