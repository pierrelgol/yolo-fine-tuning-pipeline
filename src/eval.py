from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from ultralytics import YOLO

from src.common import child_run_name, latest_train_run_name, sanitized_name
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
    selected_dataset_yaml_path = dataset_yaml_path or Path(config.evaluate.dataset_yaml)
    if not selected_dataset_yaml_path.is_absolute():
        selected_dataset_yaml_path = (config.paths.project_root / selected_dataset_yaml_path).resolve()

    selected_weights_path = weights_path or config.paths.train_best_weights_path
    if not selected_dataset_yaml_path.exists():
        raise FileNotFoundError(f"Evaluation dataset config not found: {selected_dataset_yaml_path}")
    if not selected_weights_path.exists():
        raise FileNotFoundError(f"Evaluation weights not found: {selected_weights_path}")

    parent_train_run_name = resolve_parent_train_run_name(config, selected_weights_path)
    run_name = child_run_name(parent_train_run_name, "eval")
    run_dir = config.paths.eval_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)

    configure_tracking_environment(config)
    tracking_session = start_tracking_run(
        config=config,
        task_name="eval",
        run_name=run_name,
        group_name=parent_train_run_name,
        resume="allow",
        run_config={
            "task": "eval",
            "run_name": run_name,
            "parent_train_run_name": parent_train_run_name,
            "dataset_yaml_path": str(selected_dataset_yaml_path),
            "weights_path": str(selected_weights_path),
        },
    )

    try:
        model = YOLO(str(selected_weights_path))
        evaluation_results = model.val(
            data=str(selected_dataset_yaml_path),
            project=str(config.paths.eval_dir),
            name=run_name,
            exist_ok=force,
            plots=True,
        )

        metrics_payload = dict(getattr(evaluation_results, "results_dict", {}) or {})
        metrics_json = json.dumps(metrics_payload, indent=2, sort_keys=True)
        (run_dir / "metrics.json").write_text(metrics_json, encoding="utf-8")
        config.paths.eval_latest_metrics_path.write_text(metrics_json, encoding="utf-8")

        summary_metrics = build_evaluation_summary(metrics_payload, evaluation_results)
        log_tracking_metrics(tracking_session, summary_metrics)
        log_tracking_key_value_table(tracking_session, "tables/evaluation_summary", summary_metrics)
        log_tracking_images(
            tracking_session,
            build_evaluation_image_mapping(run_dir, config.tracking.max_logged_images),
        )
        save_tracking_artifacts(
            tracking_session,
            [
                run_dir / "metrics.json",
                config.paths.eval_latest_metrics_path,
                run_dir / "args.yaml",
            ],
        )

        print(f"Evaluation run: {run_name}")
        print(f"Latest metrics: {config.paths.eval_latest_metrics_path}")
        return config.paths.eval_latest_metrics_path
    except Exception as error:
        alert_tracking_failure(tracking_session, "Evaluation failed", str(error))
        raise
    finally:
        finish_tracking_run(tracking_session)


def resolve_parent_train_run_name(config: AppConfig, weights_path: Path) -> str:
    return latest_train_run_name(
        train_dir=config.paths.train_dir,
        latest_run_path=config.paths.train_latest_run_path,
        fallback_name=sanitized_name(weights_path.stem),
    )


def configure_tracking_environment(config: AppConfig) -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("ULTRALYTICS_WANDB", "False")
    os.environ.setdefault("TRACKIO_PROJECT", config.tracking.project_name)


def build_evaluation_summary(metrics_payload: dict[str, float], evaluation_results) -> dict[str, float]:
    speed_payload = getattr(evaluation_results, "speed", {}) or {}
    return {
        "eval/precision": metrics_payload.get("metrics/precision(B)"),
        "eval/recall": metrics_payload.get("metrics/recall(B)"),
        "eval/map50": metrics_payload.get("metrics/mAP50(B)"),
        "eval/map50_95": metrics_payload.get("metrics/mAP50-95(B)"),
        "eval/fitness": metrics_payload.get("fitness"),
        "eval/speed_preprocess_ms": speed_payload.get("preprocess"),
        "eval/speed_inference_ms": speed_payload.get("inference"),
        "eval/speed_loss_ms": speed_payload.get("loss"),
        "eval/speed_postprocess_ms": speed_payload.get("postprocess"),
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
        if not image_path.exists():
            continue
        image_mapping[key] = (image_path, caption)

    return image_mapping
