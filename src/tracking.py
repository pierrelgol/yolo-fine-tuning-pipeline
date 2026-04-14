from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import AppConfig


@dataclass
class TrackingSession:
    project_name: str
    run_name: str


def prepare_tracking_environment(config: AppConfig) -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("ULTRALYTICS_WANDB", "False")
    os.environ.setdefault("TRACKIO_PROJECT", config.tracking.project_name)


def start_tracking_run(
    config: AppConfig,
    task_name: str,
    run_name: str,
    run_config: dict[str, Any],
    group_name: str | None = None,
    resume: str = "never",
) -> TrackingSession | None:
    if not config.tracking.enabled:
        return None

    prepare_tracking_environment(config)

    try:
        import trackio

        trackio.init(
            project=config.tracking.project_name,
            name=run_name,
            group=group_name or task_name,
            config=run_config,
            resume=resume,
            auto_log_gpu=config.tracking.auto_log_gpu,
            gpu_log_interval=config.tracking.gpu_log_interval_seconds,
        )
    except Exception as error:
        print(f"Trackio initialization failed: {error}")
        return None

    return TrackingSession(project_name=config.tracking.project_name, run_name=run_name)


def log_tracking_metrics(session: TrackingSession | None, metrics: dict[str, Any], step: int | None = None) -> None:
    if session is None:
        return

    payload = {key: normalize_tracking_value(value) for key, value in metrics.items() if value is not None}
    if not payload:
        return

    try:
        import trackio

        trackio.log(payload, step=step)
    except Exception as error:
        print(f"Trackio metric logging failed: {error}")


def log_tracking_images(
    session: TrackingSession | None,
    image_mapping: dict[str, tuple[Path, str | None]],
    step: int | None = None,
) -> None:
    if session is None:
        return

    try:
        import trackio
    except Exception as error:
        print(f"Trackio image support unavailable: {error}")
        return

    payload: dict[str, Any] = {}
    for key, (image_path, caption) in image_mapping.items():
        if image_path.exists():
            payload[key] = trackio.Image(image_path, caption=caption)

    if not payload:
        return

    try:
        trackio.log(payload, step=step)
    except Exception as error:
        print(f"Trackio image logging failed: {error}")


def log_tracking_table(
    session: TrackingSession | None,
    table_name: str,
    columns: list[str],
    rows: list[list[Any]],
    step: int | None = None,
) -> None:
    if session is None or not rows:
        return

    try:
        import pandas
        import trackio

        normalized_rows = [[normalize_tracking_value(value) for value in row] for row in rows]
        dataframe = pandas.DataFrame(normalized_rows, columns=columns)
        trackio.log({table_name: trackio.Table(dataframe=dataframe)}, step=step)
    except Exception as error:
        print(f"Trackio table logging failed: {error}")


def log_tracking_key_value_table(
    session: TrackingSession | None,
    table_name: str,
    values: dict[str, Any],
    step: int | None = None,
) -> None:
    rows = [[key, normalize_tracking_value(value)] for key, value in values.items() if value is not None]
    log_tracking_table(session, table_name, ["name", "value"], rows, step=step)


def log_tracking_table_from_csv(
    session: TrackingSession | None,
    table_name: str,
    csv_path: Path,
    max_rows: int,
    step: int | None = None,
) -> None:
    if session is None or not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            return

        rows: list[list[Any]] = []
        for row_index, row in enumerate(reader):
            if row_index >= max_rows:
                break
            rows.append([parse_csv_value(row.get(field_name)) for field_name in reader.fieldnames])

    log_tracking_table(session, table_name, list(reader.fieldnames), rows, step=step)


def save_tracking_artifacts(session: TrackingSession | None, artifact_paths: list[Path]) -> None:
    if session is None:
        return

    existing_paths = [path for path in artifact_paths if path.exists()]
    if not existing_paths:
        return

    try:
        import trackio

        for artifact_path in existing_paths:
            trackio.save(artifact_path, project=session.project_name)
    except Exception as error:
        print(f"Trackio artifact save failed: {error}")


def alert_tracking_failure(session: TrackingSession | None, title: str, message: str) -> None:
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
    if not stripped_value:
        return None

    try:
        numeric_value = float(stripped_value)
    except ValueError:
        return stripped_value

    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def normalize_tracking_value(value: Any) -> Any:
    if type(value) in {str, int, float, bool}:
        return value

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return normalize_tracking_value(item_method())
        except Exception:
            return str(value)

    if isinstance(value, dict):
        return {str(key): normalize_tracking_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [normalize_tracking_value(item) for item in value]

    return str(value)
