from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import torch
import yaml
from ultralytics import YOLO

from src.common import (
    copy_image,
    discover_images,
    ensure_dir,
    image_label_path,
    load_class_map,
    non_empty_file,
    parse_yolo_labels,
    save_yolo_labels,
    split_items,
    stable_name,
    yolo_label_line,
)
from src.config import AppConfig
from src.tracking import (
    alert_tracking_failure,
    finish_tracking_run,
    log_tracking_metrics,
    log_tracking_metrics_from_mapping,
    log_training_history,
    save_tracking_artifacts,
    start_tracking_run,
)


def train_model(
    config: AppConfig,
    dataset_yaml_path: Path | None = None,
    model_name: str | None = None,
    epochs: int | None = None,
    image_size: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    run_name: str | None = None,
    force: bool = False,
) -> Path:
    selected_dataset_yaml_path = dataset_yaml_path or build_training_dataset(config)
    selected_model_name = model_name or config.train.model_name
    selected_epochs = epochs if epochs is not None else config.train.epochs
    selected_image_size = image_size if image_size is not None else config.train.image_size
    selected_batch_size = batch_size if batch_size is not None else config.train.batch_size
    requested_device = device or config.train.device
    selected_device = resolve_training_device(requested_device)
    selected_run_name = run_name or config.train.run_name
    hyperparameters = config.train.hyperparameters
    selected_workers = resolve_training_workers(hyperparameters.workers)

    if not selected_dataset_yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {selected_dataset_yaml_path}")

    print(training_device_summary(requested_device, selected_device))
    print(training_worker_summary(hyperparameters.workers, selected_workers))

    run_dir = ensure_dir(config.paths.train_runs_dir / selected_run_name)
    metrics_path = run_dir / "metrics.json"

    if training_run_is_complete(run_dir) and not force:
        print(f"Skipping training, completed run already exists: {run_dir}")
        print(metrics_path)
        return metrics_path

    configure_tracking_environment(config)
    tracking_session = start_tracking_run(
        config=config,
        task_name="train",
        run_name=selected_run_name,
        run_config=build_training_tracking_config(
            dataset_yaml_path=selected_dataset_yaml_path,
            model_name=selected_model_name,
            epochs=selected_epochs,
            image_size=selected_image_size,
            batch_size=selected_batch_size,
            requested_device=requested_device,
            selected_device=selected_device,
            workers=selected_workers,
            hyperparameters=hyperparameters,
        ),
    )

    try:
        model = YOLO(selected_model_name)
    except Exception as error:
        message = (
            f"Unable to initialize model '{selected_model_name}'. "
            "Install a compatible Ultralytics version or override the model in config.toml."
        )
        alert_tracking_failure(tracking_session, "Training initialization failed", message)
        finish_tracking_run(tracking_session)
        raise RuntimeError(message) from error

    try:
        should_resume_run = run_dir.exists() and any(run_dir.iterdir()) and not training_run_is_complete(run_dir) and not force
        training_results = model.train(
            data=str(selected_dataset_yaml_path),
            epochs=selected_epochs,
            imgsz=selected_image_size,
            batch=selected_batch_size,
            device=selected_device,
            project=str(config.paths.train_runs_dir),
            name=selected_run_name,
            exist_ok=force or should_resume_run,
            resume=should_resume_run,
            plots=True,
            patience=hyperparameters.patience,
            optimizer=hyperparameters.optimizer,
            lr0=hyperparameters.initial_learning_rate,
            lrf=hyperparameters.final_learning_rate_factor,
            momentum=hyperparameters.momentum,
            weight_decay=hyperparameters.weight_decay,
            warmup_epochs=hyperparameters.warmup_epochs,
            box=hyperparameters.box_loss_gain,
            cls=hyperparameters.class_loss_gain,
            dfl=hyperparameters.dfl_loss_gain,
            hsv_h=hyperparameters.hue_augmentation,
            hsv_s=hyperparameters.saturation_augmentation,
            hsv_v=hyperparameters.value_augmentation,
            degrees=hyperparameters.rotation_degrees,
            translate=hyperparameters.translation_fraction,
            scale=hyperparameters.scaling_gain,
            shear=hyperparameters.shear_degrees,
            perspective=hyperparameters.perspective_fraction,
            flipud=hyperparameters.vertical_flip_probability,
            fliplr=hyperparameters.horizontal_flip_probability,
            mosaic=hyperparameters.mosaic_probability,
            mixup=hyperparameters.mixup_probability,
            copy_paste=hyperparameters.copy_paste_probability,
            workers=selected_workers,
        )

        metrics_payload = dict(getattr(training_results, "results_dict", {}) or {})
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")

        log_training_history(tracking_session, run_dir / "results.csv")
        log_tracking_metrics(
            tracking_session,
            {
                "dataset/train_images": len(discover_images(config.paths.training_train_images_dir)),
                "dataset/val_images": len(discover_images(config.paths.training_val_images_dir)),
            },
        )
        log_tracking_metrics_from_mapping(
            tracking_session,
            metrics_payload,
            {
                "metrics/precision(B)": "summary/precision",
                "metrics/recall(B)": "summary/recall",
                "metrics/mAP50(B)": "summary/map50",
                "metrics/mAP50-95(B)": "summary/map50_95",
                "val/box_loss": "summary/val_box_loss",
                "val/cls_loss": "summary/val_class_loss",
                "val/dfl_loss": "summary/val_dfl_loss",
                "fitness": "summary/fitness",
            },
        )
        save_tracking_artifacts(
            tracking_session,
            [
                metrics_path,
                run_dir / "results.csv",
                run_dir / "weights" / "best.pt",
                config.paths.training_dataset_yaml_path,
                config.paths.training_manifest_path,
            ],
        )

        print(f"Training complete. Metrics written to {metrics_path}")
        return metrics_path
    except Exception as error:
        alert_tracking_failure(tracking_session, "Training failed", str(error))
        raise
    finally:
        finish_tracking_run(tracking_session)


def build_training_dataset(config: AppConfig) -> Path:
    if not config.paths.coco128_dir.exists():
        raise FileNotFoundError(f"Base dataset directory not found: {config.paths.coco128_dir}. Run setup before train.")

    clear_directory(config.paths.training_dir)
    ensure_dir(config.paths.training_train_images_dir)
    ensure_dir(config.paths.training_train_labels_dir)
    ensure_dir(config.paths.training_val_images_dir)
    ensure_dir(config.paths.training_val_labels_dir)
    ensure_dir(config.paths.training_dir / "predictions" / "train2017")
    ensure_dir(config.paths.training_dir / "predictions" / "val2017")

    base_class_names = load_class_names_from_dataset_yaml(config.paths.coco128_dataset_yaml_path)
    annotation_class_map = load_annotation_class_map(config)
    merged_class_names, local_to_global_class_id = merge_annotation_classes(base_class_names, annotation_class_map)

    copy_base_dataset_into_training(config)
    copy_optional_dataset_into_training(
        image_dir=config.paths.annotation_images_dir,
        label_dir=config.paths.annotation_labels_dir,
        source_prefix="annotation",
        local_to_global_class_id=local_to_global_class_id,
        destination_image_dir=config.paths.training_train_images_dir,
        destination_label_dir=config.paths.training_train_labels_dir,
    )
    copy_optional_dataset_into_training(
        image_dir=config.paths.augmented_images_dir,
        label_dir=config.paths.augmented_labels_dir,
        source_prefix="augmented",
        local_to_global_class_id=local_to_global_class_id,
        destination_image_dir=config.paths.training_train_images_dir,
        destination_label_dir=config.paths.training_train_labels_dir,
    )

    write_training_dataset_yaml(config, merged_class_names)
    write_training_manifest(config, merged_class_names)
    return config.paths.training_dataset_yaml_path


def clear_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    ensure_dir(directory)


def copy_base_dataset_into_training(config: AppConfig) -> None:
    image_paths = discover_images(config.paths.coco128_images_dir)
    train_images, validation_images = split_items(image_paths, config.setup.train_split, config.setup.random_seed)

    for image_path in train_images:
        copy_base_sample(
            image_path=image_path,
            source_image_dir=config.paths.coco128_images_dir,
            source_label_dir=config.paths.coco128_labels_dir,
            destination_image_dir=config.paths.training_train_images_dir,
            destination_label_dir=config.paths.training_train_labels_dir,
        )

    for image_path in validation_images:
        copy_base_sample(
            image_path=image_path,
            source_image_dir=config.paths.coco128_images_dir,
            source_label_dir=config.paths.coco128_labels_dir,
            destination_image_dir=config.paths.training_val_images_dir,
            destination_label_dir=config.paths.training_val_labels_dir,
        )


def copy_base_sample(
    image_path: Path,
    source_image_dir: Path,
    source_label_dir: Path,
    destination_image_dir: Path,
    destination_label_dir: Path,
) -> None:
    destination_image_path = destination_image_dir / image_path.name
    destination_label_path = destination_label_dir / f"{image_path.stem}.txt"
    copy_image(image_path, destination_image_path)

    source_label_path = image_label_path(image_path, source_image_dir, source_label_dir)
    if source_label_path.exists():
        copy_image(source_label_path, destination_label_path)
    else:
        save_yolo_labels(destination_label_path, [])


def load_annotation_class_map(config: AppConfig) -> dict[str, int]:
    if not config.paths.annotation_classes_path.exists():
        return {}
    return load_class_map(config.paths.annotation_classes_path)


def merge_annotation_classes(
    base_class_names: list[str],
    annotation_class_map: dict[str, int],
) -> tuple[list[str], dict[int, int]]:
    merged_class_names = list(base_class_names)
    local_to_global_class_id: dict[int, int] = {}

    ordered_items = sorted(annotation_class_map.items(), key=lambda item: item[1])
    for class_name, local_class_id in ordered_items:
        if class_name in merged_class_names:
            global_class_id = merged_class_names.index(class_name)
        else:
            global_class_id = len(merged_class_names)
            merged_class_names.append(class_name)
        local_to_global_class_id[local_class_id] = global_class_id

    return merged_class_names, local_to_global_class_id


def copy_optional_dataset_into_training(
    image_dir: Path,
    label_dir: Path,
    source_prefix: str,
    local_to_global_class_id: dict[int, int],
    destination_image_dir: Path,
    destination_label_dir: Path,
) -> None:
    if not image_dir.exists():
        return

    for image_path in discover_images(image_dir):
        source_label_path = image_label_path(image_path, image_dir, label_dir)
        annotations = parse_yolo_labels(source_label_path)
        if not annotations:
            continue

        image_suffix = image_path.suffix.lower() or ".jpg"
        destination_stem = stable_name(source_prefix, image_path.as_posix(), suffix="")
        destination_image_path = destination_image_dir / f"{source_prefix}_{destination_stem}{image_suffix}"
        destination_label_path = destination_label_dir / f"{source_prefix}_{destination_stem}.txt"

        copy_image(image_path, destination_image_path)
        remapped_lines: list[str] = []
        for local_class_id, bbox in annotations:
            global_class_id = local_to_global_class_id.get(local_class_id, local_class_id)
            remapped_lines.append(yolo_label_line(global_class_id, bbox))
        save_yolo_labels(destination_label_path, remapped_lines)


def load_class_names_from_dataset_yaml(dataset_yaml_path: Path) -> list[str]:
    if not dataset_yaml_path.exists():
        return []

    payload = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
    raw_names = payload.get("names", {})
    if isinstance(raw_names, dict):
        ordered_items = sorted(raw_names.items(), key=lambda item: int(item[0]))
        return [str(name) for _, name in ordered_items]
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    return []


def write_training_dataset_yaml(config: AppConfig, class_names: list[str]) -> None:
    dataset_payload = {
        "path": str(config.paths.training_dir.resolve()),
        "train": "images/train2017",
        "val": "images/val2017",
        "names": {index: name for index, name in enumerate(class_names)},
    }
    config.paths.training_dataset_yaml_path.write_text(yaml.safe_dump(dataset_payload, sort_keys=False), encoding="utf-8")


def write_training_manifest(config: AppConfig, class_names: list[str]) -> None:
    manifest = {
        "training_dir": str(config.paths.training_dir),
        "train_image_dir": str(config.paths.training_train_images_dir),
        "train_label_dir": str(config.paths.training_train_labels_dir),
        "val_image_dir": str(config.paths.training_val_images_dir),
        "val_label_dir": str(config.paths.training_val_labels_dir),
        "num_train_images": len(discover_images(config.paths.training_train_images_dir)),
        "num_val_images": len(discover_images(config.paths.training_val_images_dir)),
        "num_classes": len(class_names),
    }
    config.paths.training_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_training_tracking_config(
    dataset_yaml_path: Path,
    model_name: str,
    epochs: int,
    image_size: int,
    batch_size: int,
    requested_device: str,
    selected_device: str,
    workers: int,
    hyperparameters,
) -> dict:
    return {
        "task": "train",
        "dataset_yaml_path": str(dataset_yaml_path),
        "model_name": model_name,
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "requested_device": requested_device,
        "selected_device": selected_device,
        "workers": workers,
        "optimizer": hyperparameters.optimizer,
        "patience": hyperparameters.patience,
        "initial_learning_rate": hyperparameters.initial_learning_rate,
        "final_learning_rate_factor": hyperparameters.final_learning_rate_factor,
        "weight_decay": hyperparameters.weight_decay,
        "mosaic_probability": hyperparameters.mosaic_probability,
        "mixup_probability": hyperparameters.mixup_probability,
    }


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

    if normalized_workers == 0:
        return 0

    # Windows data-loader workers are substantially more fragile because each
    # worker starts a new Python process. Keeping this at 0 avoids the
    # MemoryError / thread startup failures seen on this project.
    return 0


def training_worker_summary(configured_workers: int, selected_workers: int) -> str:
    if configured_workers == selected_workers:
        return f"Training data loader workers: {selected_workers}"

    return (
        f"Training data loader workers: {selected_workers} "
        f"(configured={configured_workers}, reduced for Windows stability)"
    )


def training_run_is_complete(run_dir: Path) -> bool:
    weights_path = run_dir / "weights" / "best.pt"
    results_csv_path = run_dir / "results.csv"
    return non_empty_file(weights_path) and non_empty_file(results_csv_path)
