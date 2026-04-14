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
    group_name: str


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

    return TrackingSession(
        project_name=config.tracking.project_name,
        run_name=run_name,
        task_name=task_name,
        group_name=group_name or task_name,
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
        filtered_metrics[key] = normalize_tracking_value(value)

    if not filtered_metrics:
        return

    try:
        import trackio

        trackio.log(filtered_metrics, step=step)
    except Exception as error:
        print(f"Trackio logging failed: {error}")


def log_tracking_images(
    session: TrackingSession | None,
    image_mapping: dict[str, tuple[Path, str | None]],
    step: int | None = None,
) -> None:
    if session is None:
        return

    filtered_payload: dict[str, Any] = {}
    for key, (image_path, caption) in image_mapping.items():
        if not image_path.exists():
            continue

        try:
            import trackio

            filtered_payload[key] = trackio.Image(image_path, caption=caption)
        except Exception as error:
            print(f"Trackio image preparation failed for {image_path}: {error}")

    if not filtered_payload:
        return

    try:
        import trackio

        trackio.log(filtered_payload, step=step)
    except Exception as error:
        print(f"Trackio image logging failed: {error}")


def log_tracking_table(
    session: TrackingSession | None,
    table_name: str,
    columns: list[str],
    rows: list[list[Any]],
    step: int | None = None,
) -> None:
    if session is None:
        return
    if not rows:
        return

    try:
        import pandas
        import trackio

        normalized_rows = []
        for row in rows:
            normalized_rows.append([normalize_tracking_value(value) for value in row])

        dataframe = pandas.DataFrame(normalized_rows, columns=columns)
        table = trackio.Table(dataframe=dataframe)
        trackio.log({table_name: table}, step=step)
    except Exception as error:
        print(f"Trackio table logging failed: {error}")


def log_tracking_key_value_table(
    session: TrackingSession | None,
    table_name: str,
    values: dict[str, Any],
    step: int | None = None,
) -> None:
    rows: list[list[Any]] = []
    for key, value in values.items():
        if value is None:
            continue
        rows.append([key, normalize_tracking_value(value)])

    log_tracking_table(
        session=session,
        table_name=table_name,
        columns=["name", "value"],
        rows=rows,
        step=step,
    )


def log_tracking_table_from_csv(
    session: TrackingSession | None,
    table_name: str,
    csv_path: Path,
    max_rows: int,
    step: int | None = None,
) -> None:
    if session is None:
        return
    if not csv_path.exists():
        return

    with csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if reader.fieldnames is None:
            return

        table_rows: list[list[Any]] = []
        for row_index, row in enumerate(reader):
            if row_index >= max_rows:
                break

            table_row: list[Any] = []
            for field_name in reader.fieldnames:
                raw_value = row.get(field_name)
                table_row.append(normalize_tracking_value(parse_csv_value(raw_value)))
            table_rows.append(table_row)

    log_tracking_table(
        session=session,
        table_name=table_name,
        columns=[field_name.strip() for field_name in reader.fieldnames],
        rows=table_rows,
        step=step,
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


def normalize_tracking_value(value: Any) -> Any:
    if type(value) in {str, int, float, bool}:
        return value

    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return normalize_tracking_value(item_method())
        except Exception:
            return str(value)

    if isinstance(value, list):
        return [normalize_tracking_value(item) for item in value]

    if isinstance(value, tuple):
        return [normalize_tracking_value(item) for item in value]

    if isinstance(value, dict):
        normalized_mapping: dict[str, Any] = {}
        for key, item in value.items():
            normalized_mapping[str(key)] = normalize_tracking_value(item)
        return normalized_mapping

    return str(value)
