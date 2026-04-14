from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

from ultralytics import YOLO

from src.common import (
    child_run_name,
    discover_images,
    ensure_dir,
    latest_train_run_name,
    resolve_dataset_directory,
    sanitized_name,
    save_yolo_labels,
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
    selected_dataset_path = dataset_subdir or Path(config.infer.dataset_name)
    selected_dataset_dir = resolve_dataset_directory(
        project_root=config.paths.project_root,
        dataset_root=config.paths.dataset_dir,
        dataset_path=selected_dataset_path,
    )
    selected_weights_path = weights_path or config.paths.train_best_weights_path
    if not selected_weights_path.exists():
        raise FileNotFoundError(f"Inference weights not found: {selected_weights_path}")

    images_root = resolve_inference_image_root(selected_dataset_dir)
    predictions_root = prediction_output_directory(selected_dataset_dir, images_root)
    source_image_paths = discover_images(images_root)
    if not source_image_paths:
        raise FileNotFoundError(f"No images found in {images_root}")

    parent_train_run_name = resolve_parent_train_run_name(config, selected_weights_path)
    run_name = child_run_name(parent_train_run_name, "infer")
    run_dir = config.paths.infer_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    clear_prediction_directory(selected_dataset_dir)
    ensure_dir(predictions_root)

    configure_tracking_environment(config)
    tracking_session = start_tracking_run(
        config=config,
        task_name="infer",
        run_name=run_name,
        group_name=parent_train_run_name,
        resume="allow",
        run_config={
            "task": "infer",
            "run_name": run_name,
            "parent_train_run_name": parent_train_run_name,
            "dataset_dir": str(selected_dataset_dir),
            "images_root": str(images_root),
            "weights_path": str(selected_weights_path),
            "num_images": len(source_image_paths),
        },
    )

    try:
        model = YOLO(str(selected_weights_path))
        prediction_results = model.predict(
            source=[str(path) for path in source_image_paths],
            project=str(config.paths.infer_dir),
            name=run_name,
            exist_ok=force,
            save=True,
            save_txt=False,
        )

        write_prediction_files(source_image_paths, images_root, predictions_root, prediction_results)

        summary_metrics = summarize_inference_results(prediction_results)
        latest_manifest = write_inference_manifests(
            config=config,
            run_dir=run_dir,
            selected_dataset_dir=selected_dataset_dir,
            selected_weights_path=selected_weights_path,
            predictions_root=predictions_root,
            run_name=run_name,
            num_images=len(source_image_paths),
            images_root=images_root,
        )

        log_tracking_metrics(tracking_session, summary_metrics)
        log_tracking_key_value_table(tracking_session, "tables/inference_summary", summary_metrics)
        log_tracking_table(
            tracking_session,
            "tables/inference_samples",
            columns=["image", "num_detections", "top_class_id", "top_confidence", "average_confidence"],
            rows=build_inference_sample_rows(source_image_paths, prediction_results, config.tracking.max_logged_table_rows),
        )
        log_tracking_images(
            tracking_session,
            build_inference_image_mapping(run_dir, config.tracking.max_logged_images),
        )
        save_tracking_artifacts(
            tracking_session,
            [
                latest_manifest,
                selected_dataset_dir / "predictions_manifest.json",
            ],
        )

        print(f"Inference run: {run_name}")
        print(f"Predictions: {predictions_root}")
        return predictions_root
    except Exception as error:
        alert_tracking_failure(tracking_session, "Inference failed", str(error))
        raise
    finally:
        finish_tracking_run(tracking_session)


def configure_tracking_environment(config: AppConfig) -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("ULTRALYTICS_WANDB", "False")
    os.environ.setdefault("TRACKIO_PROJECT", config.tracking.project_name)


def resolve_parent_train_run_name(config: AppConfig, weights_path: Path) -> str:
    return latest_train_run_name(
        train_dir=config.paths.train_dir,
        latest_run_path=config.paths.train_latest_run_path,
        fallback_name=sanitized_name(weights_path.stem),
    )


def resolve_inference_image_root(dataset_dir: Path) -> Path:
    val_image_dir = dataset_dir / "images" / "val2017"
    if discover_images(val_image_dir):
        return val_image_dir
    return dataset_dir / "images"


def prediction_output_directory(dataset_dir: Path, images_root: Path) -> Path:
    base_images_dir = dataset_dir / "images"
    base_predictions_dir = dataset_dir / "predictions"

    if images_root == base_images_dir:
        return base_predictions_dir

    relative_split = images_root.relative_to(base_images_dir)
    return base_predictions_dir / relative_split


def clear_prediction_directory(dataset_dir: Path) -> None:
    predictions_dir = dataset_dir / "predictions"
    if predictions_dir.exists():
        shutil.rmtree(predictions_dir)


def write_prediction_files(
    image_paths: list[Path],
    images_root: Path,
    predictions_root: Path,
    prediction_results: list,
) -> None:
    for image_path, prediction_result in zip(image_paths, prediction_results, strict=True):
        relative_image_path = image_path.relative_to(images_root)
        prediction_path = predictions_root / relative_image_path.with_suffix(".txt")
        prediction_lines = prediction_lines_for_result(prediction_result)
        save_yolo_labels(prediction_path, prediction_lines)


def prediction_lines_for_result(prediction_result) -> list[str]:
    boxes = getattr(prediction_result, "boxes", None)
    if boxes is None:
        return []

    class_tensor = boxes.cls
    bbox_tensor = boxes.xywhn
    if class_tensor is None or bbox_tensor is None:
        return []

    prediction_lines: list[str] = []
    for index in range(len(boxes)):
        class_id = int(class_tensor[index].item())
        bbox_values = bbox_tensor[index].tolist()
        prediction_lines.append(yolo_label_line(class_id, bbox_values))

    return prediction_lines


def write_inference_manifests(
    config: AppConfig,
    run_dir: Path,
    selected_dataset_dir: Path,
    selected_weights_path: Path,
    predictions_root: Path,
    run_name: str,
    num_images: int,
    images_root: Path,
) -> Path:
    manifest = {
        "run_name": run_name,
        "dataset_dir": str(selected_dataset_dir),
        "images_root": str(images_root),
        "predictions_dir": str(predictions_root),
        "run_dir": str(run_dir),
        "weights_path": str(selected_weights_path),
        "num_images": num_images,
    }

    run_manifest_path = run_dir / "manifest.json"
    dataset_manifest_path = selected_dataset_dir / "predictions_manifest.json"

    run_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dataset_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config.paths.infer_latest_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return config.paths.infer_latest_manifest_path


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

        if boxes is not None and num_detections > 0:
            confidence_tensor = boxes.conf
            class_tensor = boxes.cls
            if confidence_tensor is not None:
                confidence_values = [float(confidence_value) for confidence_value in confidence_tensor.tolist()]
                top_confidence = max(confidence_values)
                average_confidence = sum(confidence_values) / len(confidence_values)

                if class_tensor is not None:
                    top_index = int(confidence_tensor.argmax().item())
                    top_class_id = int(class_tensor[top_index].item())

        rows.append([image_path.name, num_detections, top_class_id, top_confidence, average_confidence])

    return rows


def build_inference_image_mapping(run_dir: Path, max_logged_images: int) -> dict[str, tuple[Path, str | None]]:
    rendered_image_paths = discover_images(run_dir)
    image_mapping: dict[str, tuple[Path, str | None]] = {}

    for index, image_path in enumerate(rendered_image_paths):
        if index >= max_logged_images:
            break
        image_mapping[f"images/inference_sample_{index + 1}"] = (image_path, f"Rendered output for {image_path.name}.")

    return image_mapping


def summarize_inference_results(prediction_results: list) -> dict[str, float]:
    num_images = len(prediction_results)
    total_detections = 0
    images_with_detections = 0
    confidence_values: list[float] = []
    inference_times: list[float] = []
    preprocess_times: list[float] = []
    postprocess_times: list[float] = []

    for prediction_result in prediction_results:
        boxes = getattr(prediction_result, "boxes", None)
        num_detections = len(boxes) if boxes is not None else 0
        total_detections += num_detections

        if num_detections > 0:
            images_with_detections += 1
            confidence_tensor = boxes.conf
            if confidence_tensor is not None:
                for confidence_value in confidence_tensor.tolist():
                    confidence_values.append(float(confidence_value))

        speed_payload = getattr(prediction_result, "speed", {}) or {}
        if "preprocess" in speed_payload:
            preprocess_times.append(float(speed_payload["preprocess"]))
        if "inference" in speed_payload:
            inference_times.append(float(speed_payload["inference"]))
        if "postprocess" in speed_payload:
            postprocess_times.append(float(speed_payload["postprocess"]))

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
