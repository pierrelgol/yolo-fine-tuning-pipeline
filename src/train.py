from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import torch
import yaml
from ultralytics import YOLO

from src.common import (
    copy_image,
    discover_images,
    ensure_dir,
    image_label_path,
    load_class_map,
    next_run_name,
    non_empty_file,
    ordered_class_names,
    parse_yolo_labels,
    save_yolo_labels,
    write_json,
)
from src.config import AppConfig
from src.tracking import (
    alert_tracking_failure,
    finish_tracking_run,
    log_tracking_images,
    log_tracking_key_value_table,
    log_tracking_metrics,
    log_tracking_table_from_csv,
    save_tracking_artifacts,
    start_tracking_run,
)


@dataclass(frozen=True)
class DatasetSample:
    image_path: Path
    label_path: Path | None
    source_name: str


def train_model(
    config: AppConfig,
    dataset_yaml_path: Path | None = None,
    model_name: str | None = None,
    epochs: int | None = None,
    image_size: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    force: bool = False,
) -> Path:
    selected_model_name = model_name or config.train.model_name
    selected_epochs = epochs if epochs is not None else config.train.epochs
    selected_image_size = image_size if image_size is not None else config.train.image_size
    selected_batch_size = batch_size if batch_size is not None else config.train.batch_size
    requested_device = device or config.train.device
    selected_device = resolve_training_device(requested_device)
    selected_workers = resolve_training_workers(config.train.hyperparameters.workers)

    selected_dataset_yaml_path = dataset_yaml_path or build_training_dataset(config)
    if not selected_dataset_yaml_path.exists():
        raise FileNotFoundError(f"Training dataset config not found: {selected_dataset_yaml_path}")

    selected_run_name = next_run_name(config.paths.run_versions_path, selected_model_name)
    run_dir = config.paths.train_runs_dir / selected_run_name
    if run_dir.exists() and not force:
        raise FileExistsError(f"Training run already exists: {run_dir}")

    print(training_device_summary(requested_device, selected_device))
    print(training_worker_summary(config.train.hyperparameters.workers, selected_workers))
    print(f"Training run: {selected_run_name}")

    configure_tracking_environment(config)
    tracking_session = start_tracking_run(
        config=config,
        task_name="train",
        run_name=selected_run_name,
        group_name=selected_run_name,
        run_config=build_tracking_config(
            config=config,
            run_name=selected_run_name,
            model_name=selected_model_name,
            dataset_yaml_path=selected_dataset_yaml_path,
            epochs=selected_epochs,
            image_size=selected_image_size,
            batch_size=selected_batch_size,
            requested_device=requested_device,
            selected_device=selected_device,
            workers=selected_workers,
        ),
    )

    try:
        model = YOLO(selected_model_name)
        register_training_callbacks(
            model=model,
            tracking_session=tracking_session,
            log_every_n_steps=config.tracking.log_every_n_steps,
        )

        training_results = model.train(
            data=str(selected_dataset_yaml_path),
            epochs=selected_epochs,
            imgsz=selected_image_size,
            batch=selected_batch_size,
            device=selected_device,
            project=str(config.paths.train_runs_dir),
            name=selected_run_name,
            exist_ok=force,
            plots=True,
            save=True,
            patience=config.train.hyperparameters.patience,
            optimizer=config.train.hyperparameters.optimizer,
            lr0=config.train.hyperparameters.initial_learning_rate,
            lrf=config.train.hyperparameters.final_learning_rate_factor,
            momentum=config.train.hyperparameters.momentum,
            weight_decay=config.train.hyperparameters.weight_decay,
            warmup_epochs=config.train.hyperparameters.warmup_epochs,
            box=config.train.hyperparameters.box_loss_gain,
            cls=config.train.hyperparameters.class_loss_gain,
            dfl=config.train.hyperparameters.dfl_loss_gain,
            hsv_h=config.train.hyperparameters.hue_augmentation,
            hsv_s=config.train.hyperparameters.saturation_augmentation,
            hsv_v=config.train.hyperparameters.value_augmentation,
            degrees=config.train.hyperparameters.rotation_degrees,
            translate=config.train.hyperparameters.translation_fraction,
            scale=config.train.hyperparameters.scaling_gain,
            shear=config.train.hyperparameters.shear_degrees,
            perspective=config.train.hyperparameters.perspective_fraction,
            flipud=config.train.hyperparameters.vertical_flip_probability,
            fliplr=config.train.hyperparameters.horizontal_flip_probability,
            mosaic=config.train.hyperparameters.mosaic_probability,
            mixup=config.train.hyperparameters.mixup_probability,
            copy_paste=config.train.hyperparameters.copy_paste_probability,
            workers=selected_workers,
        )

        metrics_payload = dict(getattr(training_results, "results_dict", {}) or {})
        write_training_outputs(config, run_dir, metrics_payload)
        write_latest_training_run_metadata(
            config=config,
            run_name=selected_run_name,
            model_name=selected_model_name,
            run_dir=run_dir,
            dataset_yaml_path=selected_dataset_yaml_path,
        )
        log_training_summary(config, tracking_session, run_dir, metrics_payload)

        print(f"Training dataset: {config.paths.train_dir}")
        print(f"Best weights: {config.paths.train_best_weights_path}")
        print(f"Latest weights: {config.paths.train_latest_weights_path}")
        return config.paths.train_metrics_path
    except Exception as error:
        alert_tracking_failure(tracking_session, "Training failed", str(error))
        raise
    finally:
        finish_tracking_run(tracking_session)


def build_training_dataset(config: AppConfig) -> Path:
    class_map = load_class_map(config.paths.annotation_classes_path)
    class_names = ordered_class_names(class_map)
    if not class_names:
        raise FileNotFoundError(
            f"No annotation classes found in {config.paths.annotation_classes_path}. Run annotate before train."
        )

    dataset_samples = collect_training_samples(config)
    if not dataset_samples:
        raise FileNotFoundError("No training images found. Run annotate or augment before train.")

    if count_positive_samples(dataset_samples) == 0:
        raise ValueError("No labeled annotations found for training. Add at least one bounding box before train.")

    clear_training_data(config)
    ensure_dir(config.paths.train_train_images_dir)
    ensure_dir(config.paths.train_train_labels_dir)
    ensure_dir(config.paths.train_val_images_dir)
    ensure_dir(config.paths.train_val_labels_dir)

    train_samples, val_samples = split_training_samples(
        samples=dataset_samples,
        train_ratio=config.setup.train_split,
        random_seed=config.setup.random_seed,
    )

    valid_class_ids = set(class_map.values())
    copy_training_split(train_samples, config.paths.train_train_images_dir, config.paths.train_train_labels_dir, valid_class_ids)
    copy_training_split(val_samples, config.paths.train_val_images_dir, config.paths.train_val_labels_dir, valid_class_ids)

    write_training_dataset_yaml(config, class_names)
    write_training_manifest(config, class_names, train_samples, val_samples)
    return config.paths.train_dataset_yaml_path


def collect_training_samples(config: AppConfig) -> list[DatasetSample]:
    samples: list[DatasetSample] = []
    samples.extend(collect_samples_from_directory(config.paths.annotation_images_dir, config.paths.annotation_labels_dir, "annotation"))
    samples.extend(
        collect_samples_from_directory(
            config.paths.augmented_train_images_dir,
            config.paths.augmented_train_labels_dir,
            "augmented",
        )
    )
    return samples


def collect_samples_from_directory(image_dir: Path, label_dir: Path, source_name: str) -> list[DatasetSample]:
    if not image_dir.exists():
        return []

    samples: list[DatasetSample] = []
    for image_path in discover_images(image_dir):
        label_path = image_label_path(image_path, image_dir, label_dir)
        if label_path.exists():
            sample_label_path: Path | None = label_path
        else:
            sample_label_path = None

        samples.append(
            DatasetSample(
                image_path=image_path,
                label_path=sample_label_path,
                source_name=source_name,
            )
        )

    return samples


def count_positive_samples(samples: list[DatasetSample]) -> int:
    positive_sample_count = 0
    for sample in samples:
        if sample.label_path is None:
            continue
        if parse_yolo_labels(sample.label_path):
            positive_sample_count += 1
    return positive_sample_count


def clear_training_data(config: AppConfig) -> None:
    remove_directory_if_present(config.paths.train_data_dir)
    remove_file_if_present(config.paths.train_dataset_yaml_path)
    remove_file_if_present(config.paths.train_manifest_path)


def split_training_samples(
    samples: list[DatasetSample],
    train_ratio: float,
    random_seed: int,
) -> tuple[list[DatasetSample], list[DatasetSample]]:
    if len(samples) == 1:
        only_sample = samples[0]
        return [only_sample], [only_sample]

    shuffled_samples = list(samples)
    random_generator = random.Random(random_seed)
    random_generator.shuffle(shuffled_samples)

    split_index = int(len(shuffled_samples) * train_ratio)
    split_index = max(1, split_index)
    split_index = min(len(shuffled_samples) - 1, split_index)

    train_samples = sorted(shuffled_samples[:split_index], key=lambda sample: sample.image_path.name)
    val_samples = sorted(shuffled_samples[split_index:], key=lambda sample: sample.image_path.name)
    return train_samples, val_samples


def copy_training_split(
    samples: list[DatasetSample],
    destination_image_dir: Path,
    destination_label_dir: Path,
    valid_class_ids: set[int],
) -> None:
    for sample in samples:
        destination_image_path = destination_image_dir / sample.image_path.name
        destination_label_path = destination_label_dir / f"{sample.image_path.stem}.txt"

        copy_image(sample.image_path, destination_image_path)
        label_lines = build_filtered_label_lines(sample.label_path, valid_class_ids)
        save_yolo_labels(destination_label_path, label_lines)


def build_filtered_label_lines(label_path: Path | None, valid_class_ids: set[int]) -> list[str]:
    if label_path is None or not label_path.exists():
        return []

    filtered_lines: list[str] = []
    for class_id, bbox in parse_yolo_labels(label_path):
        if class_id not in valid_class_ids:
            continue
        x_center, y_center, width, height = bbox
        filtered_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return filtered_lines


def write_training_dataset_yaml(config: AppConfig, class_names: list[str]) -> None:
    payload = {
        "path": str(config.paths.train_data_dir.resolve()),
        "train": "images/train2017",
        "val": "images/val2017",
        "names": {index: class_name for index, class_name in enumerate(class_names)},
    }
    config.paths.train_dataset_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_training_manifest(
    config: AppConfig,
    class_names: list[str],
    train_samples: list[DatasetSample],
    val_samples: list[DatasetSample],
) -> None:
    manifest = {
        "dataset_dir": str(config.paths.train_dir),
        "data_dir": str(config.paths.train_data_dir),
        "train_image_dir": str(config.paths.train_train_images_dir),
        "train_label_dir": str(config.paths.train_train_labels_dir),
        "val_image_dir": str(config.paths.train_val_images_dir),
        "val_label_dir": str(config.paths.train_val_labels_dir),
        "classes": class_names,
        "num_train_images": len(train_samples),
        "num_val_images": len(val_samples),
    }
    config.paths.train_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def register_training_callbacks(
    model: YOLO,
    tracking_session,
    log_every_n_steps: int,
) -> None:
    callback_state = {"global_step": 0}

    def on_train_batch_end(trainer) -> None:
        callback_state["global_step"] += 1
        global_step = callback_state["global_step"]
        if global_step % log_every_n_steps != 0:
            return

        batch_metrics = trainer.label_loss_items(trainer.tloss)
        batch_metrics["train/epoch"] = trainer.epoch + 1
        batch_metrics["optimizer/lr"] = float(trainer.optimizer.param_groups[0]["lr"])
        log_tracking_metrics(tracking_session, batch_metrics, step=global_step)

    def on_fit_epoch_end(trainer) -> None:
        epoch_metrics = dict(getattr(trainer, "metrics", {}) or {})
        epoch_metrics["train/epoch"] = trainer.epoch + 1
        epoch_metrics["train/fitness"] = getattr(trainer, "fitness", None)
        learning_rates = getattr(trainer, "lr", {}) or {}
        if "lr/pg0" in learning_rates:
            epoch_metrics["optimizer/lr_group0"] = learning_rates["lr/pg0"]
        log_tracking_metrics(tracking_session, epoch_metrics, step=callback_state["global_step"])

    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def log_training_summary(
    config: AppConfig,
    tracking_session,
    run_dir: Path,
    metrics_payload: dict[str, Any],
) -> None:
    log_tracking_metrics(
        tracking_session,
        {
            "dataset/train_images": len(discover_images(config.paths.train_train_images_dir)),
            "dataset/val_images": len(discover_images(config.paths.train_val_images_dir)),
        },
    )
    log_tracking_key_value_table(
        tracking_session,
        "tables/training_summary",
        build_training_summary(metrics_payload),
    )
    log_tracking_table_from_csv(
        tracking_session,
        "tables/training_history",
        config.paths.train_results_csv_path,
        max_rows=config.tracking.max_logged_table_rows,
    )
    log_tracking_images(
        tracking_session,
        build_training_image_mapping(run_dir, config.tracking.max_logged_images),
    )
    save_tracking_artifacts(
        tracking_session,
        [
            config.paths.train_metrics_path,
            config.paths.train_results_csv_path,
            config.paths.train_best_weights_path,
            config.paths.train_latest_weights_path,
            config.paths.train_latest_run_path,
            config.paths.train_dataset_yaml_path,
            config.paths.train_manifest_path,
            run_dir / "metrics.json",
        ],
    )


def write_training_outputs(config: AppConfig, run_dir: Path, metrics_payload: dict[str, Any]) -> None:
    ensure_dir(config.paths.train_dir)
    metrics_json = json.dumps(metrics_payload, indent=2, sort_keys=True)
    config.paths.train_metrics_path.write_text(metrics_json, encoding="utf-8")
    (run_dir / "metrics.json").write_text(metrics_json, encoding="utf-8")

    copy_required_file(run_dir / "results.csv", config.paths.train_results_csv_path)
    copy_required_file(run_dir / "weights" / "best.pt", config.paths.train_best_weights_path)
    copy_required_file(run_dir / "weights" / "last.pt", config.paths.train_latest_weights_path)


def write_latest_training_run_metadata(
    config: AppConfig,
    run_name: str,
    model_name: str,
    run_dir: Path,
    dataset_yaml_path: Path,
) -> None:
    payload = {
        "run_name": run_name,
        "model_name": model_name,
        "run_dir": str(run_dir),
        "dataset_yaml_path": str(dataset_yaml_path),
        "best_weights_path": str(config.paths.train_best_weights_path),
        "latest_weights_path": str(config.paths.train_latest_weights_path),
    }
    write_json(config.paths.train_latest_run_path, payload)


def build_tracking_config(
    config: AppConfig,
    run_name: str,
    model_name: str,
    dataset_yaml_path: Path,
    epochs: int,
    image_size: int,
    batch_size: int,
    requested_device: str,
    selected_device: str,
    workers: int,
) -> dict[str, Any]:
    return {
        "task": "train",
        "run_name": run_name,
        "model_name": model_name,
        "dataset_yaml_path": str(dataset_yaml_path),
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "requested_device": requested_device,
        "selected_device": selected_device,
        "workers": workers,
        "log_every_n_steps": config.tracking.log_every_n_steps,
    }


def build_training_summary(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision": metrics_payload.get("metrics/precision(B)"),
        "recall": metrics_payload.get("metrics/recall(B)"),
        "map50": metrics_payload.get("metrics/mAP50(B)"),
        "map50_95": metrics_payload.get("metrics/mAP50-95(B)"),
        "val_box_loss": metrics_payload.get("val/box_loss"),
        "val_class_loss": metrics_payload.get("val/cls_loss"),
        "val_dfl_loss": metrics_payload.get("val/dfl_loss"),
        "fitness": metrics_payload.get("fitness"),
    }


def build_training_image_mapping(run_dir: Path, max_logged_images: int) -> dict[str, tuple[Path, str | None]]:
    candidate_images = [
        ("images/training_curves", run_dir / "results.png", "Training curves."),
        ("images/confusion_matrix", run_dir / "confusion_matrix.png", "Validation confusion matrix."),
        ("images/precision_recall_curve", run_dir / "PR_curve.png", "Precision recall curve."),
        ("images/train_batch_preview", run_dir / "train_batch0.jpg", "Training batch preview."),
        ("images/val_prediction_preview", run_dir / "val_batch0_pred.jpg", "Validation prediction preview."),
    ]

    image_mapping: dict[str, tuple[Path, str | None]] = {}
    for key, image_path, caption in candidate_images:
        if len(image_mapping) >= max_logged_images:
            break
        if not image_path.exists():
            continue
        image_mapping[key] = (image_path, caption)

    return image_mapping


def configure_tracking_environment(config: AppConfig) -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("ULTRALYTICS_WANDB", "False")
    os.environ.setdefault("TRACKIO_PROJECT", config.tracking.project_name)


def resolve_training_device(requested_device: str) -> str:
    normalized_device = requested_device.strip().lower()
    if normalized_device == "":
        return "cpu"
    if normalized_device == "cpu":
        return "cpu"
    if normalized_device == "auto":
        if torch.cuda.is_available():
            return "0"
        return "cpu"

    cuda_like_requests = {"cuda", "cuda:0", "0"}
    if normalized_device in cuda_like_requests:
        if torch.cuda.is_available():
            return "0"
        raise RuntimeError(build_cuda_unavailable_message(requested_device))

    if normalized_device.startswith("cuda"):
        if torch.cuda.is_available():
            return normalized_device.replace("cuda:", "")
        raise RuntimeError(build_cuda_unavailable_message(requested_device))

    return requested_device


def training_device_summary(requested_device: str, selected_device: str) -> str:
    if selected_device == "cpu":
        return (
            f"Training device: cpu "
            f"(requested={requested_device!r}, torch_cuda_available={torch.cuda.is_available()})"
        )

    device_index = int(selected_device)
    device_name = torch.cuda.get_device_name(device_index)
    return f"Training device: cuda:{device_index} ({device_name})"


def build_cuda_unavailable_message(requested_device: str) -> str:
    return (
        f"GPU training was requested with device={requested_device!r}, "
        "but this Python environment cannot see CUDA.\n"
        f"torch version: {torch.__version__}\n"
        f"torch.cuda.is_available(): {torch.cuda.is_available()}\n"
        f"torch.cuda.device_count(): {torch.cuda.device_count()}\n"
        "Install a CUDA-enabled PyTorch build in this project's environment, then run training again."
    )


def resolve_training_workers(configured_workers: int) -> int:
    normalized_workers = max(0, int(configured_workers))
    if sys.platform != "win32":
        return normalized_workers
    return 0


def training_worker_summary(configured_workers: int, selected_workers: int) -> str:
    if configured_workers == selected_workers:
        return f"Training data loader workers: {selected_workers}"

    return (
        f"Training data loader workers: {selected_workers} "
        f"(configured={configured_workers}, reduced for Windows stability)"
    )


def copy_required_file(source_path: Path, destination_path: Path) -> None:
    if not non_empty_file(source_path):
        raise FileNotFoundError(f"Expected training output not found: {source_path}")
    copy_image(source_path, destination_path)


def remove_directory_if_present(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)


def remove_file_if_present(path: Path) -> None:
    if path.exists():
        path.unlink()
