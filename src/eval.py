from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO

from src.common import child_run_name, latest_train_run_name, portable_path, remove_path, resolve_path, sanitized_name
from src.config import AppConfig
from src.tracking import (
    alert_tracking_failure,
    finish_tracking_run,
    log_tracking_images,
    log_tracking_key_value_table,
    log_tracking_metrics,
    save_tracking_artifacts,
    start_tracking_run,
)


def evaluate_model(
    config: AppConfig,
    dataset_yaml_path: Path | None = None,
    weights_path: Path | None = None,
    force: bool = False,
) -> Path:
    selected_dataset_yaml_path = resolve_dataset_yaml_path(config, dataset_yaml_path)
    selected_weights_path = (
        resolve_path(weights_path, base_dir=config.paths.project_root)
        if weights_path is not None
        else config.paths.train_best_weights_path
    )
    if not selected_dataset_yaml_path.exists():
        raise FileNotFoundError(f"Evaluation dataset config not found: {selected_dataset_yaml_path}")
    if not selected_weights_path.exists():
        raise FileNotFoundError(f"Evaluation weights not found: {selected_weights_path}")

    parent_run_name = latest_train_run_name(
        config.paths.train_runs_dir,
        config.paths.train_latest_run_path,
        sanitized_name(selected_weights_path.stem),
    )
    run_name = child_run_name(parent_run_name, "eval")
    run_dir = config.paths.eval_runs_dir / run_name
    remove_path(run_dir)

    tracking_session = start_tracking_run(
        config=config,
        task_name="eval",
        run_name=run_name,
        group_name=parent_run_name,
        resume="allow",
        run_config={
            "task": "eval",
            "run_name": run_name,
            "parent_train_run_name": parent_run_name,
            "dataset_yaml_path": portable_path(selected_dataset_yaml_path, base_dir=config.paths.project_root),
            "weights_path": portable_path(selected_weights_path, base_dir=config.paths.project_root),
        },
    )

    try:
        model = YOLO(str(selected_weights_path))
        evaluation_results = model.val(
            data=str(selected_dataset_yaml_path),
            project=str(config.paths.eval_runs_dir),
            name=run_name,
            exist_ok=True,
            plots=True,
        )

        metrics = dict(getattr(evaluation_results, "results_dict", {}) or {})
        metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
        (run_dir / "metrics.json").write_text(metrics_json, encoding="utf-8")
        config.paths.eval_latest_metrics_path.write_text(metrics_json, encoding="utf-8")

        summary = build_evaluation_summary(metrics, evaluation_results)
        log_tracking_metrics(tracking_session, summary)
        log_tracking_key_value_table(tracking_session, "tables/evaluation_summary", summary)
        log_tracking_images(tracking_session, build_evaluation_image_mapping(run_dir, config.tracking.max_logged_images))
        save_tracking_artifacts(
            tracking_session,
            [run_dir / "metrics.json", config.paths.eval_latest_metrics_path, run_dir / "args.yaml"],
        )

        print(f"Evaluation run: {run_name}")
        print(f"Latest metrics: {config.paths.eval_latest_metrics_path}")
        return config.paths.eval_latest_metrics_path
    except Exception as error:
        alert_tracking_failure(tracking_session, "Evaluation failed", str(error))
        raise
    finally:
        finish_tracking_run(tracking_session)


def resolve_dataset_yaml_path(config: AppConfig, dataset_yaml_path: Path | None) -> Path:
    selected_dataset_yaml_path = dataset_yaml_path or Path(config.evaluate.dataset_yaml)
    return resolve_path(selected_dataset_yaml_path, base_dir=config.paths.project_root)


def build_evaluation_summary(metrics: dict[str, float], evaluation_results) -> dict[str, float]:
    speed = getattr(evaluation_results, "speed", {}) or {}
    return {
        "eval/precision": metrics.get("metrics/precision(B)"),
        "eval/recall": metrics.get("metrics/recall(B)"),
        "eval/map50": metrics.get("metrics/mAP50(B)"),
        "eval/map50_95": metrics.get("metrics/mAP50-95(B)"),
        "eval/fitness": metrics.get("fitness"),
        "eval/speed_preprocess_ms": speed.get("preprocess"),
        "eval/speed_inference_ms": speed.get("inference"),
        "eval/speed_loss_ms": speed.get("loss"),
        "eval/speed_postprocess_ms": speed.get("postprocess"),
    }


def build_evaluation_image_mapping(run_dir: Path, max_logged_images: int) -> dict[str, tuple[Path, str | None]]:
    candidate_images = [
        ("images/eval_confusion_matrix", run_dir / "confusion_matrix.png", "Evaluation confusion matrix."),
        ("images/eval_precision_recall_curve", run_dir / "PR_curve.png", "Evaluation precision recall curve."),
        ("images/eval_prediction_preview", run_dir / "val_batch0_pred.jpg", "Evaluation prediction preview."),
    ]

    image_mapping: dict[str, tuple[Path, str | None]] = {}
    for key, image_path, caption in candidate_images:
        if len(image_mapping) >= max_logged_images:
            break
        if image_path.exists():
            image_mapping[key] = (image_path, caption)
    return image_mapping
