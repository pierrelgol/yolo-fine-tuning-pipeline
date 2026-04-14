from __future__ import annotations

import json
from pathlib import Path
import shutil

from ultralytics import YOLO

from src.common import discover_images, ensure_dir, resolve_dataset_directory, save_yolo_labels, yolo_label_line
from src.config import AppConfig
from src.tracking import (
    alert_tracking_failure,
    finish_tracking_run,
    log_tracking_metrics,
    save_tracking_artifacts,
    start_tracking_run,
)


def run_inference(
    config: AppConfig,
    dataset_subdir: Path,
    weights_path: Path | None = None,
    run_name: str | None = None,
    force: bool = False,
) -> Path:
    selected_dataset_dir = resolve_dataset_directory(
        project_root=config.paths.project_root,
        dataset_root=config.paths.dataset_dir,
        dataset_path=dataset_subdir,
    )
    selected_weights_path = weights_path or default_weights_path(config)
    selected_run_name = run_name or config.infer.run_name

    if not selected_weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {selected_weights_path}")

    images_root = selected_dataset_dir / "images"
    predictions_root = selected_dataset_dir / "predictions"
    if not images_root.exists():
        raise FileNotFoundError(f"Expected images directory not found: {images_root}")

    source_image_paths = discover_images(images_root)
    if not source_image_paths:
        raise FileNotFoundError(f"No images found in {images_root}")

    prediction_files = list(predictions_root.rglob("*.txt")) if predictions_root.exists() else []
    if prediction_files and not force:
        print(f"Skipping inference, predictions already exist under {predictions_root}")
        return predictions_root

    if predictions_root.exists():
        shutil.rmtree(predictions_root)
    ensure_dir(predictions_root)

    run_dir = config.paths.infer_runs_dir / selected_run_name
    if force and run_dir.exists():
        shutil.rmtree(run_dir)

    tracking_session = start_tracking_run(
        config=config,
        task_name="infer",
        run_name=selected_run_name,
        run_config={
            "task": "infer",
            "dataset_dir": str(selected_dataset_dir),
            "weights_path": str(selected_weights_path),
            "run_name": selected_run_name,
            "num_images": len(source_image_paths),
        },
    )

    try:
        model = YOLO(str(selected_weights_path))
        prediction_results = model.predict(
            source=[str(path) for path in source_image_paths],
            project=str(config.paths.infer_runs_dir),
            name=selected_run_name,
            exist_ok=True,
            save=True,
            save_txt=False,
        )

        write_prediction_files(
            image_paths=source_image_paths,
            images_root=images_root,
            predictions_root=predictions_root,
            prediction_results=prediction_results,
        )
        write_prediction_manifest(
            dataset_dir=selected_dataset_dir,
            weights_path=selected_weights_path,
            run_dir=run_dir,
            predictions_root=predictions_root,
            num_images=len(source_image_paths),
        )
        log_tracking_metrics(tracking_session, summarize_inference_results(prediction_results))
        save_tracking_artifacts(
            tracking_session,
            [
                selected_dataset_dir / "predictions_manifest.json",
            ],
        )

        print(f"Predictions written to {predictions_root}")
        print(f"Rendered inference images written to {run_dir}")
        return predictions_root
    except Exception as error:
        alert_tracking_failure(tracking_session, "Inference failed", str(error))
        raise
    finally:
        finish_tracking_run(tracking_session)


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


def write_prediction_manifest(
    dataset_dir: Path,
    weights_path: Path,
    run_dir: Path,
    predictions_root: Path,
    num_images: int,
) -> None:
    manifest_path = dataset_dir / "predictions_manifest.json"
    manifest = {
        "dataset_dir": str(dataset_dir),
        "predictions_dir": str(predictions_root),
        "run_dir": str(run_dir),
        "weights_path": str(weights_path),
        "num_images": num_images,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def default_weights_path(config: AppConfig) -> Path:
    return config.paths.train_runs_dir / config.train.run_name / "weights" / "best.pt"


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
        if "inference" in speed_payload:
            inference_times.append(float(speed_payload["inference"]))
        if "preprocess" in speed_payload:
            preprocess_times.append(float(speed_payload["preprocess"]))
        if "postprocess" in speed_payload:
            postprocess_times.append(float(speed_payload["postprocess"]))

    average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    average_detections = total_detections / num_images if num_images else 0.0
    average_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
    average_preprocess_time = sum(preprocess_times) / len(preprocess_times) if preprocess_times else 0.0
    average_postprocess_time = sum(postprocess_times) / len(postprocess_times) if postprocess_times else 0.0

    return {
        "infer/num_images": num_images,
        "infer/images_with_detections": images_with_detections,
        "infer/total_detections": total_detections,
        "infer/average_detections_per_image": average_detections,
        "infer/average_confidence": average_confidence,
        "infer/max_confidence": max(confidence_values) if confidence_values else 0.0,
        "infer/min_confidence": min(confidence_values) if confidence_values else 0.0,
        "infer/speed_preprocess_ms": average_preprocess_time,
        "infer/speed_inference_ms": average_inference_time,
        "infer/speed_postprocess_ms": average_postprocess_time,
    }
