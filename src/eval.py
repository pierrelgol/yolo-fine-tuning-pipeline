from __future__ import annotations

import json
import os
from pathlib import Path

from ultralytics import YOLO

from src.common import ensure_dir
from src.config import AppConfig
from src.tracking import (
    alert_tracking_failure,
    finish_tracking_run,
    log_tracking_metrics,
    log_tracking_metrics_from_mapping,
    save_tracking_artifacts,
    start_tracking_run,
)


def evaluate_model(
    config: AppConfig,
    dataset_yaml_path: Path | None = None,
    weights_path: Path | None = None,
    run_name: str | None = None,
    force: bool = False,
) -> Path:
    selected_dataset_yaml_path = dataset_yaml_path or config.paths.training_dataset_yaml_path
    selected_weights_path = weights_path or default_weights_path(config)
    selected_run_name = run_name or config.evaluate.run_name

    if not selected_dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {selected_dataset_yaml_path}")
    if not selected_weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {selected_weights_path}")

    run_dir = ensure_dir(config.paths.eval_runs_dir / selected_run_name)
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists() and not force:
        print(f"Skipping evaluation, metrics already exist: {metrics_path}")
        print(metrics_path)
        return metrics_path

    configure_tracking_environment(config)
    tracking_session = start_tracking_run(
        config=config,
        task_name="eval",
        run_name=selected_run_name,
        run_config={
            "task": "eval",
            "dataset_yaml_path": str(selected_dataset_yaml_path),
            "weights_path": str(selected_weights_path),
            "run_name": selected_run_name,
        },
    )

    try:
        model = YOLO(str(selected_weights_path))
        evaluation_results = model.val(
            data=str(selected_dataset_yaml_path),
            project=str(config.paths.eval_runs_dir),
            name=selected_run_name,
            exist_ok=force,
        )
        metrics_payload = dict(getattr(evaluation_results, "results_dict", {}) or {})
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

        log_tracking_metrics_from_mapping(
            tracking_session,
            metrics_payload,
            {
                "metrics/precision(B)": "eval/precision",
                "metrics/recall(B)": "eval/recall",
                "metrics/mAP50(B)": "eval/map50",
                "metrics/mAP50-95(B)": "eval/map50_95",
                "fitness": "eval/fitness",
            },
        )
        log_tracking_metrics(tracking_session, extract_speed_metrics(evaluation_results))
        save_tracking_artifacts(
            tracking_session,
            [
                metrics_path,
                run_dir / "args.yaml",
            ],
        )

        print(metrics_path)
        return metrics_path
    except Exception as error:
        alert_tracking_failure(tracking_session, "Evaluation failed", str(error))
        raise
    finally:
        finish_tracking_run(tracking_session)


def default_weights_path(config: AppConfig) -> Path:
    return config.paths.train_runs_dir / config.train.run_name / "weights" / "best.pt"


def configure_tracking_environment(config: AppConfig) -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("ULTRALYTICS_WANDB", "False")
    os.environ.setdefault("TRACKIO_PROJECT", config.tracking.project_name)


def extract_speed_metrics(evaluation_results) -> dict[str, float]:
    speed_payload = getattr(evaluation_results, "speed", {}) or {}
    metrics: dict[str, float] = {}
    for source_key, target_key in {
        "preprocess": "eval/speed_preprocess_ms",
        "inference": "eval/speed_inference_ms",
        "loss": "eval/speed_loss_ms",
        "postprocess": "eval/speed_postprocess_ms",
    }.items():
        if source_key in speed_payload:
            metrics[target_key] = float(speed_payload[source_key])
    return metrics
