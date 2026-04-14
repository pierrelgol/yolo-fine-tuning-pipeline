from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import AppConfig


@dataclass
class TrackingSession:
    project_name: str
    run_name: str
    task_name: str


def start_tracking_run(
    config: AppConfig,
    task_name: str,
    run_name: str,
    run_config: dict[str, Any],
) -> TrackingSession | None:
    if not config.tracking.enabled:
        return None

    try:
        import trackio

        trackio.init(
            project=config.tracking.project_name,
            name=run_name,
            group=task_name,
            config=run_config,
            auto_log_gpu=config.tracking.auto_log_gpu,
            gpu_log_interval=config.tracking.gpu_log_interval_seconds,
        )
    except Exception as error:
        print(f"Trackio initialization failed: {error}")
        return None

    return TrackingSession(
        project_name=config.tracking.project_name,
        run_name=run_name,
        task_name=task_name,
    )


def log_tracking_metrics(
    session: TrackingSession | None,
    metrics: dict[str, Any],
    step: int | None = None,
) -> None:
    if session is None:
        return

    filtered_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        filtered_metrics[key] = value

    if not filtered_metrics:
        return

    try:
        import trackio

        trackio.log(filtered_metrics, step=step)
    except Exception as error:
        print(f"Trackio logging failed: {error}")


def log_tracking_metrics_from_mapping(
    session: TrackingSession | None,
    source_metrics: dict[str, Any],
    metric_mapping: dict[str, str],
    step: int | None = None,
) -> None:
    selected_metrics: dict[str, Any] = {}
    for source_key, target_key in metric_mapping.items():
        if source_key not in source_metrics:
            continue
        selected_metrics[target_key] = source_metrics[source_key]

    log_tracking_metrics(session, selected_metrics, step=step)


def log_training_history(
    session: TrackingSession | None,
    results_csv_path: Path,
) -> None:
    if session is None:
        return
    if not results_csv_path.exists():
        return

    metric_mapping = {
        "train/box_loss": "train/box_loss",
        "train/cls_loss": "train/class_loss",
        "train/dfl_loss": "train/dfl_loss",
        "metrics/precision(B)": "train/precision",
        "metrics/recall(B)": "train/recall",
        "metrics/mAP50(B)": "train/map50",
        "metrics/mAP50-95(B)": "train/map50_95",
        "val/box_loss": "validation/box_loss",
        "val/cls_loss": "validation/class_loss",
        "val/dfl_loss": "validation/dfl_loss",
        "lr/pg0": "optimizer/lr_group0",
    }

    with results_csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            cleaned_row = {key.strip(): parse_csv_value(value) for key, value in row.items() if key is not None}
            epoch_value = cleaned_row.get("epoch")
            if epoch_value is None:
                continue

            epoch_index = int(float(epoch_value))
            log_tracking_metrics_from_mapping(
                session=session,
                source_metrics=cleaned_row,
                metric_mapping=metric_mapping,
                step=epoch_index + 1,
            )


def save_tracking_artifacts(
    session: TrackingSession | None,
    artifact_paths: list[Path],
) -> None:
    if session is None:
        return

    existing_artifact_paths = [path for path in artifact_paths if path.exists()]
    if not existing_artifact_paths:
        return

    try:
        import trackio

        for artifact_path in existing_artifact_paths:
            trackio.save(artifact_path, project=session.project_name)
    except Exception as error:
        print(f"Trackio artifact save failed: {error}")


def alert_tracking_failure(
    session: TrackingSession | None,
    title: str,
    message: str,
) -> None:
    if session is None:
        return

    try:
        import trackio

        trackio.alert(title=title, text=message, level="error")
    except Exception as error:
        print(f"Trackio alert failed: {error}")


def finish_tracking_run(session: TrackingSession | None) -> None:
    if session is None:
        return

    try:
        import trackio

        trackio.finish()
        trackio.context_vars.current_run.set(None)
    except Exception as error:
        print(f"Trackio finish failed: {error}")


def parse_csv_value(raw_value: str | None) -> Any:
    if raw_value is None:
        return None

    stripped_value = raw_value.strip()
    if stripped_value == "":
        return None

    try:
        numeric_value = float(stripped_value)
    except ValueError:
        return stripped_value

    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value
