from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.common import (
    child_run_name,
    dataset_predictions_dir,
    discover_images,
    latest_train_run_name,
    preferred_image_split,
    remove_path,
    resolve_dataset_directory,
    sanitized_name,
    save_yolo_labels,
    write_json,
    yolo_label_line,
)
from src.config import AppConfig
from src.tracking import (
    alert_tracking_failure,
    finish_tracking_run,
    log_tracking_images,
    log_tracking_key_value_table,
    log_tracking_metrics,
    log_tracking_table,
    save_tracking_artifacts,
    start_tracking_run,
)


def run_inference(
    config: AppConfig,
    dataset_subdir: Path | None = None,
    weights_path: Path | None = None,
    force: bool = False,
) -> Path:
    dataset_path = dataset_subdir or Path(config.infer.dataset_name)
    dataset_dir = resolve_dataset_directory(config.paths.project_root, config.paths.dataset_dir, dataset_path)
    weights_file_path = weights_path or config.paths.train_best_weights_path
    if not weights_file_path.exists():
        raise FileNotFoundError(f"Inference weights not found: {weights_file_path}")

    image_split = preferred_image_split(dataset_dir)
    if not image_split:
        raise FileNotFoundError(f"No images found in {dataset_dir / 'images'}")

    images_root = dataset_dir / "images" / image_split
    predictions_root = dataset_predictions_dir(dataset_dir, image_split)
    source_image_paths = discover_images(images_root)

    parent_run_name = latest_train_run_name(
        config.paths.train_runs_dir,
        config.paths.train_latest_run_path,
        sanitized_name(weights_file_path.stem),
    )
    run_name = child_run_name(parent_run_name, "infer")
    run_dir = config.paths.infer_runs_dir / run_name
    remove_path(run_dir)
    remove_path(dataset_dir / "predictions")

    tracking_session = start_tracking_run(
        config=config,
        task_name="infer",
        run_name=run_name,
        group_name=parent_run_name,
        resume="allow",
        run_config={
            "task": "infer",
            "run_name": run_name,
            "parent_train_run_name": parent_run_name,
            "dataset_dir": str(dataset_dir),
            "images_root": str(images_root),
            "weights_path": str(weights_file_path),
            "num_images": len(source_image_paths),
        },
    )

    try:
        model = YOLO(str(weights_file_path))
        prediction_results = model.predict(
            source=[str(path) for path in source_image_paths],
            project=str(config.paths.infer_runs_dir),
            name=run_name,
            exist_ok=True,
            save=True,
            save_txt=False,
        )

        for image_path, prediction_result in zip(source_image_paths, prediction_results, strict=True):
            prediction_path = predictions_root / image_path.relative_to(images_root).with_suffix(".txt")
            save_yolo_labels(prediction_path, prediction_lines_for_result(prediction_result))

        summary = summarize_inference_results(prediction_results)
        manifest = {
            "run_name": run_name,
            "dataset_dir": str(dataset_dir),
            "image_split": image_split,
            "images_root": str(images_root),
            "predictions_dir": str(predictions_root),
            "run_dir": str(run_dir),
            "weights_path": str(weights_file_path),
            "num_images": len(source_image_paths),
        }
        write_json(run_dir / "manifest.json", manifest)
        write_json(dataset_dir / "predictions_manifest.json", manifest)
        write_json(config.paths.infer_latest_manifest_path, manifest)

        log_tracking_metrics(tracking_session, summary)
        log_tracking_key_value_table(tracking_session, "tables/inference_summary", summary)
        log_tracking_table(
            tracking_session,
            "tables/inference_samples",
            ["image", "num_detections", "top_class_id", "top_confidence", "average_confidence"],
            build_inference_sample_rows(source_image_paths, prediction_results, config.tracking.max_logged_table_rows),
        )
        log_tracking_images(tracking_session, build_inference_image_mapping(run_dir, config.tracking.max_logged_images))
        save_tracking_artifacts(
            tracking_session,
            [run_dir / "manifest.json", dataset_dir / "predictions_manifest.json", config.paths.infer_latest_manifest_path],
        )

        print(f"Inference run: {run_name}")
        print(f"Predictions: {predictions_root}")
        return predictions_root
    except Exception as error:
        alert_tracking_failure(tracking_session, "Inference failed", str(error))
        raise
    finally:
        finish_tracking_run(tracking_session)


def prediction_lines_for_result(prediction_result) -> list[str]:
    boxes = getattr(prediction_result, "boxes", None)
    if boxes is None or boxes.cls is None or boxes.xywhn is None:
        return []

    prediction_lines: list[str] = []
    for index in range(len(boxes)):
        prediction_lines.append(yolo_label_line(int(boxes.cls[index].item()), boxes.xywhn[index].tolist()))
    return prediction_lines


def build_inference_sample_rows(
    image_paths: list[Path],
    prediction_results: list,
    max_rows: int,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for image_path, prediction_result in zip(image_paths, prediction_results, strict=True):
        if len(rows) >= max_rows:
            break

        boxes = getattr(prediction_result, "boxes", None)
        num_detections = len(boxes) if boxes is not None else 0
        top_class_id: int | None = None
        top_confidence: float | None = None
        average_confidence: float | None = None

        if boxes is not None and num_detections > 0 and boxes.conf is not None:
            confidence_values = [float(value) for value in boxes.conf.tolist()]
            top_confidence = max(confidence_values)
            average_confidence = sum(confidence_values) / len(confidence_values)
            if boxes.cls is not None:
                top_index = int(boxes.conf.argmax().item())
                top_class_id = int(boxes.cls[top_index].item())

        rows.append([image_path.name, num_detections, top_class_id, top_confidence, average_confidence])
    return rows


def build_inference_image_mapping(run_dir: Path, max_logged_images: int) -> dict[str, tuple[Path, str | None]]:
    image_mapping: dict[str, tuple[Path, str | None]] = {}
    for index, image_path in enumerate(discover_images(run_dir)):
        if index >= max_logged_images:
            break
        image_mapping[f"images/inference_sample_{index + 1}"] = (image_path, f"Rendered output for {image_path.name}.")
    return image_mapping


def summarize_inference_results(prediction_results: list) -> dict[str, float]:
    num_images = len(prediction_results)
    total_detections = 0
    images_with_detections = 0
    confidence_values: list[float] = []
    preprocess_times: list[float] = []
    inference_times: list[float] = []
    postprocess_times: list[float] = []

    for prediction_result in prediction_results:
        boxes = getattr(prediction_result, "boxes", None)
        num_detections = len(boxes) if boxes is not None else 0
        total_detections += num_detections

        if num_detections > 0:
            images_with_detections += 1
            if boxes.conf is not None:
                confidence_values.extend(float(value) for value in boxes.conf.tolist())

        speed = getattr(prediction_result, "speed", {}) or {}
        if "preprocess" in speed:
            preprocess_times.append(float(speed["preprocess"]))
        if "inference" in speed:
            inference_times.append(float(speed["inference"]))
        if "postprocess" in speed:
            postprocess_times.append(float(speed["postprocess"]))

    average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    average_detections = total_detections / num_images if num_images else 0.0

    return {
        "infer/num_images": num_images,
        "infer/images_with_detections": images_with_detections,
        "infer/total_detections": total_detections,
        "infer/average_detections_per_image": average_detections,
        "infer/average_confidence": average_confidence,
        "infer/max_confidence": max(confidence_values) if confidence_values else 0.0,
        "infer/min_confidence": min(confidence_values) if confidence_values else 0.0,
        "infer/speed_preprocess_ms": average(preprocess_times),
        "infer/speed_inference_ms": average(inference_times),
        "infer/speed_postprocess_ms": average(postprocess_times),
    }


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
