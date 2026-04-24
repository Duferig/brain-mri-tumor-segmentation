from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class RuntimePaths:
    root_dir: Path = ROOT_DIR
    artifacts_dir: Path = ROOT_DIR / "artifacts"
    predictions_dir: Path = ROOT_DIR / "artifacts" / "predictions"
    uploads_dir: Path = ROOT_DIR / "artifacts" / "uploads"
    metrics_dir: Path = ROOT_DIR / "artifacts" / "metrics"
    models_dir: Path = ROOT_DIR / "artifacts" / "models"
    manifests_dir: Path = ROOT_DIR / "artifacts" / "manifests"
    runs_dir: Path = ROOT_DIR / "artifacts" / "runs"

    def ensure(self) -> None:
        for path in (
            self.artifacts_dir,
            self.predictions_dir,
            self.uploads_dir,
            self.metrics_dir,
            self.models_dir,
            self.manifests_dir,
            self.runs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("rb") as file_obj:
        return tomllib.load(file_obj)


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    base = root or ROOT_DIR
    return (base / candidate).resolve()


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
