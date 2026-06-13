from __future__ import annotations

import json
from pathlib import Path

import joblib

from mltrainer.core.model_registry import OUTPUTS_DIR, get_dataset_spec


def dataset_output_dir(dataset: str) -> Path:
    spec = get_dataset_spec(dataset)
    return OUTPUTS_DIR / spec.name


def metrics_path(dataset: str) -> Path:
    return dataset_output_dir(dataset) / "metrics.json"


def best_model_path(dataset: str) -> Path:
    return dataset_output_dir(dataset) / "best_model.joblib"


def write_training_outputs(dataset: str, metrics: dict[str, object], model: object) -> tuple[Path, Path]:
    output_dir = dataset_output_dir(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path(dataset)
    model_file = best_model_path(dataset)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, model_file)
    return metrics_file, model_file


def read_metrics(dataset: str) -> dict[str, object]:
    metrics_file = metrics_path(dataset)
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Metrics file not found for '{dataset}': {metrics_file}. Run 'mltrainer train --dataset {dataset}' first."
        )
    return json.loads(metrics_file.read_text(encoding="utf-8"))


def update_metrics(dataset: str, metrics: dict[str, object]) -> Path:
    metrics_file = metrics_path(dataset)
    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Metrics file not found for '{dataset}': {metrics_file}. Run 'mltrainer train --dataset {dataset}' first."
        )
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics_file


def overwrite_best_model(dataset: str, model: object) -> Path:
    model_file = best_model_path(dataset)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_file)
    return model_file
