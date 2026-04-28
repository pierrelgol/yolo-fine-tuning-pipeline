from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from src.common import (
    TRAIN_SPLIT,
    VAL_SPLIT,
    copy_file,
    dataset_classes_path,
    dataset_images_dir,
    dataset_labels_dir,
    dataset_manifest_path,
    dataset_yaml_path,
    discover_images,
    image_label_path,
    load_class_map,
    next_run_name,
    non_empty_file,
    ordered_class_names,
    parse_yolo_labels,
    portable_path,
    remove_path,
    resolve_path,
    save_class_map,
    save_yolo_labels,
    write_dataset_yaml,
    write_json,
    yolo_label_line,
)
from src.config import AppConfig, CurriculumConfig, NumericRangeConfig
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


@dataclass(frozen=True)
class CurriculumPhase:
    stage_index: int
    phase_name: str
    difficulty: float
    epochs: int
    train_kwargs: dict[str, Any]
    augmentations: list[Any]


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
    selected_epochs = (
        config.train.curriculum.main_epochs_per_stage
        if epochs is None
        else epochs
    )
    if selected_epochs < 0:
        raise ValueError("--epochs must be zero or greater")
    selected_image_size = (
        config.train.image_size if image_size is None else image_size
    )
    selected_batch_size = (
        config.train.batch_size if batch_size is None else batch_size
    )
    requested_device = device or config.train.device
    selected_device = resolve_training_device(requested_device)
    selected_workers = resolve_training_workers(
        config.train.hyperparameters.workers
    )

    selected_dataset_yaml_path = (
        resolve_path(dataset_yaml_path, base_dir=config.paths.project_root)
        if dataset_yaml_path is not None
        else build_training_dataset(config)
    )
    if not selected_dataset_yaml_path.exists():
        raise FileNotFoundError(
            f"Training dataset config not found: {selected_dataset_yaml_path}"
        )

    run_name = next_run_name(
        config.paths.run_versions_path, selected_model_name
    )
    run_dir = config.paths.train_runs_dir / run_name
    if run_dir.exists() and not force:
        raise FileExistsError(f"Training run already exists: {run_dir}")

    print(training_device_summary(requested_device, selected_device))
    print(
        training_worker_summary(
            config.train.hyperparameters.workers, selected_workers
        )
    )
    print(f"Training run: {run_name}")

    tracking_session = start_tracking_run(
        config=config,
        task_name="train",
        run_name=run_name,
        group_name=run_name,
        run_config={
            "task": "train",
            "run_name": run_name,
            "model_name": selected_model_name,
            "dataset_yaml_path": portable_path(
                selected_dataset_yaml_path, base_dir=config.paths.project_root
            ),
            "main_epochs_per_stage": selected_epochs,
            "curriculum_stages": config.train.curriculum.stages,
            "image_size": selected_image_size,
            "batch_size": selected_batch_size,
            "requested_device": requested_device,
            "selected_device": selected_device,
            "workers": selected_workers,
            "log_every_n_steps": config.tracking.log_every_n_steps,
        },
    )

    try:
        curriculum_phases = build_curriculum_phases(
            config.train.curriculum,
            main_epochs_per_stage=selected_epochs,
        )
        phase_runs: list[dict[str, Any]] = []
        current_model_name = selected_model_name
        training_results = None
        final_phase_run_dir = run_dir

        for phase_number, phase in enumerate(curriculum_phases, start=1):
            phase_run_name = (
                f"{run_name}-s{phase.stage_index + 1:02d}-{phase.phase_name}"
            )
            phase_run_dir = config.paths.train_runs_dir / phase_run_name
            print(
                f"Curriculum phase {phase_number}/{len(curriculum_phases)}: "
                f"stage={phase.stage_index + 1}, phase={phase.phase_name}, "
                f"difficulty={phase.difficulty:.3f}, epochs={phase.epochs}"
            )
            print(f"  Ultralytics kwargs: {phase.train_kwargs}")
            print(
                "  Albumentations: "
                + ", ".join(
                    str(transform) for transform in phase.augmentations
                )
            )

            model = YOLO(current_model_name)
            register_training_callbacks(
                model, tracking_session, config.tracking.log_every_n_steps
            )
            training_results = model.train(
                data=str(selected_dataset_yaml_path),
                epochs=phase.epochs,
                imgsz=selected_image_size,
                batch=selected_batch_size,
                device=selected_device,
                project=str(config.paths.train_runs_dir),
                name=phase_run_name,
                exist_ok=force,
                plots=True,
                save=True,
                patience=config.train.hyperparameters.patience,
                optimizer=config.train.hyperparameters.optimizer,
                lr0=config.train.hyperparameters.initial_learning_rate,
                lrf=config.train.hyperparameters.final_learning_rate_factor,
                momentum=config.train.hyperparameters.momentum,
                weight_decay=config.train.hyperparameters.weight_decay,
                warmup_epochs=min(
                    config.train.hyperparameters.warmup_epochs,
                    float(phase.epochs),
                ),
                box=config.train.hyperparameters.box_loss_gain,
                cls=config.train.hyperparameters.class_loss_gain,
                dfl=config.train.hyperparameters.dfl_loss_gain,
                close_mosaic=config.train.hyperparameters.close_mosaic,
                copy_paste_mode=config.train.hyperparameters.copy_paste_mode,
                auto_augment=config.train.hyperparameters.auto_augment,
                erasing=config.train.hyperparameters.erasing,
                workers=selected_workers,
                **phase.train_kwargs,
                augmentations=phase.augmentations,
            )
            current_model_name = str(phase_run_dir / "weights" / "last.pt")
            final_phase_run_dir = phase_run_dir
            phase_runs.append(
                {
                    "stage": phase.stage_index + 1,
                    "phase": phase.phase_name,
                    "difficulty": phase.difficulty,
                    "epochs": phase.epochs,
                    "run_name": phase_run_name,
                    "run_dir": portable_path(
                        phase_run_dir, base_dir=config.paths.project_root
                    ),
                    "kwargs": phase.train_kwargs,
                    "augmentations": [
                        str(transform) for transform in phase.augmentations
                    ],
                }
            )

        metrics = dict(getattr(training_results, "results_dict", {}) or {})
        write_training_outputs(config, final_phase_run_dir, run_dir, metrics)
        write_json(
            config.paths.train_latest_run_path,
            {
                "run_name": run_name,
                "model_name": selected_model_name,
                "run_dir": portable_path(
                    run_dir, base_dir=config.paths.project_root
                ),
                "dataset_yaml_path": portable_path(
                    selected_dataset_yaml_path,
                    base_dir=config.paths.project_root,
                ),
                "best_weights_path": portable_path(
                    config.paths.train_best_weights_path,
                    base_dir=config.paths.project_root,
                ),
                "latest_weights_path": portable_path(
                    config.paths.train_latest_weights_path,
                    base_dir=config.paths.project_root,
                ),
                "curriculum_phases": phase_runs,
            },
        )
        log_training_summary(
            config, tracking_session, final_phase_run_dir, metrics
        )

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
    class_map = load_class_map(
        dataset_classes_path(config.paths.augmented_dir)
    )
    class_names = ordered_class_names(class_map)
    if not class_names:
        raise FileNotFoundError(
            f"No classes found in augmented dataset. Run augment before train."
        )

    valid_class_ids = set(class_map.values())
    augmented_train_samples = collect_samples(
        config.paths.augmented_dir, TRAIN_SPLIT
    )
    augmented_val_samples = collect_samples(
        config.paths.augmented_dir, VAL_SPLIT
    )

    if augmented_train_samples:
        train_samples = augmented_train_samples
        val_samples = augmented_val_samples if augmented_val_samples else list(augmented_train_samples)
    else:
        raise FileNotFoundError(
            "No augmented training images found. Run augment before train."
        )

    if count_positive_samples(train_samples + val_samples) == 0:
        raise ValueError(
            "No labeled samples found for training. Check that augment produced labeled images."
        )

    clear_training_dataset(config)
    train_image_dir = dataset_images_dir(config.paths.train_dir, TRAIN_SPLIT)
    train_label_dir = dataset_labels_dir(config.paths.train_dir, TRAIN_SPLIT)
    val_image_dir = dataset_images_dir(config.paths.train_dir, VAL_SPLIT)
    val_label_dir = dataset_labels_dir(config.paths.train_dir, VAL_SPLIT)

    train_samples = sorted(
        train_samples, key=lambda sample: sample.image_path.name
    )
    val_samples = sorted(val_samples, key=lambda sample: sample.image_path.name)
    copy_split(train_samples, train_image_dir, train_label_dir, valid_class_ids)
    copy_split(val_samples, val_image_dir, val_label_dir, valid_class_ids)

    save_class_map(dataset_classes_path(config.paths.train_dir), class_map)
    write_dataset_yaml(config.paths.train_dir, class_names)
    write_json(
        dataset_manifest_path(config.paths.train_dir),
        {
            "dataset_dir": portable_path(
                config.paths.train_dir, base_dir=config.paths.project_root
            ),
            "train_image_dir": portable_path(
                train_image_dir, base_dir=config.paths.project_root
            ),
            "train_label_dir": portable_path(
                train_label_dir, base_dir=config.paths.project_root
            ),
            "val_image_dir": portable_path(
                val_image_dir, base_dir=config.paths.project_root
            ),
            "val_label_dir": portable_path(
                val_label_dir, base_dir=config.paths.project_root
            ),
            "classes": class_names,
            "num_train_images": len(train_samples),
            "num_val_images": len(val_samples),
        },
    )
    return dataset_yaml_path(config.paths.train_dir)


def collect_samples(dataset_dir: Path, split: str) -> list[DatasetSample]:
    image_dir = dataset_images_dir(dataset_dir, split)
    label_dir = dataset_labels_dir(dataset_dir, split)
    if not image_dir.exists():
        return []

    samples: list[DatasetSample] = []
    for image_path in discover_images(image_dir):
        label_path = image_label_path(image_path, image_dir, label_dir)
        samples.append(
            DatasetSample(
                image_path=image_path,
                label_path=label_path if label_path.exists() else None,
                source_name=dataset_dir.name,
            )
        )
    return samples


def count_positive_samples(samples: list[DatasetSample]) -> int:
    positive_sample_count = 0
    for sample in samples:
        if sample.label_path and parse_yolo_labels(sample.label_path):
            positive_sample_count += 1
    return positive_sample_count


def clear_training_dataset(config: AppConfig) -> None:
    for path in [
        config.paths.train_dir / "images",
        config.paths.train_dir / "labels",
        dataset_yaml_path(config.paths.train_dir),
        dataset_classes_path(config.paths.train_dir),
        dataset_manifest_path(config.paths.train_dir),
        config.paths.train_metrics_path,
        config.paths.train_results_csv_path,
        config.paths.train_best_weights_path,
        config.paths.train_latest_weights_path,
    ]:
        remove_path(path)

    config.paths.train_dir.mkdir(parents=True, exist_ok=True)


def copy_split(
    samples: list[DatasetSample],
    destination_image_dir: Path,
    destination_label_dir: Path,
    valid_class_ids: set[int],
) -> None:
    for sample in samples:
        destination_image_path = destination_image_dir / sample.image_path.name
        destination_label_path = (
            destination_label_dir / f"{sample.image_path.stem}.txt"
        )
        copy_file(sample.image_path, destination_image_path)

        label_lines: list[str] = []
        if sample.label_path and sample.label_path.exists():
            for class_id, bbox in parse_yolo_labels(sample.label_path):
                if class_id in valid_class_ids:
                    label_lines.append(yolo_label_line(class_id, bbox))
        save_yolo_labels(destination_label_path, label_lines)


def register_training_callbacks(
    model: YOLO, tracking_session, log_every_n_steps: int
) -> None:
    callback_state = {"global_step": 0}

    def on_train_batch_end(trainer) -> None:
        callback_state["global_step"] += 1
        global_step = callback_state["global_step"]
        if global_step % log_every_n_steps != 0:
            return

        batch_metrics = trainer.label_loss_items(trainer.tloss)
        batch_metrics["train/epoch"] = trainer.epoch + 1
        batch_metrics["optimizer/lr"] = float(
            trainer.optimizer.param_groups[0]["lr"]
        )
        log_tracking_metrics(tracking_session, batch_metrics, step=global_step)

    def on_fit_epoch_end(trainer) -> None:
        epoch_metrics = dict(getattr(trainer, "metrics", {}) or {})
        epoch_metrics["train/epoch"] = trainer.epoch + 1
        epoch_metrics["train/fitness"] = getattr(trainer, "fitness", None)
        learning_rates = getattr(trainer, "lr", {}) or {}
        if "lr/pg0" in learning_rates:
            epoch_metrics["optimizer/lr_group0"] = learning_rates["lr/pg0"]
        log_tracking_metrics(
            tracking_session, epoch_metrics, step=callback_state["global_step"]
        )

    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def write_training_outputs(
    config: AppConfig,
    source_run_dir: Path,
    summary_run_dir: Path,
    metrics: dict[str, Any],
) -> None:
    summary_run_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = json.dumps(metrics, indent=2, sort_keys=True)
    config.paths.train_metrics_path.write_text(metrics_json, encoding="utf-8")
    (summary_run_dir / "metrics.json").write_text(
        metrics_json, encoding="utf-8"
    )
    (source_run_dir / "metrics.json").write_text(metrics_json, encoding="utf-8")
    copy_output_file(
        source_run_dir / "results.csv", config.paths.train_results_csv_path
    )
    copy_output_file(
        source_run_dir / "weights" / "best.pt",
        config.paths.train_best_weights_path,
    )
    copy_output_file(
        source_run_dir / "weights" / "last.pt",
        config.paths.train_latest_weights_path,
    )


def log_training_summary(
    config: AppConfig, tracking_session, run_dir: Path, metrics: dict[str, Any]
) -> None:
    log_tracking_metrics(
        tracking_session,
        {
            "dataset/train_images": len(
                discover_images(
                    dataset_images_dir(config.paths.train_dir, TRAIN_SPLIT)
                )
            ),
            "dataset/val_images": len(
                discover_images(
                    dataset_images_dir(config.paths.train_dir, VAL_SPLIT)
                )
            ),
        },
    )
    log_tracking_key_value_table(
        tracking_session,
        "tables/training_summary",
        {
            "precision": metrics.get("metrics/precision(B)"),
            "recall": metrics.get("metrics/recall(B)"),
            "map50": metrics.get("metrics/mAP50(B)"),
            "map50_95": metrics.get("metrics/mAP50-95(B)"),
            "val_box_loss": metrics.get("val/box_loss"),
            "val_class_loss": metrics.get("val/cls_loss"),
            "val_dfl_loss": metrics.get("val/dfl_loss"),
            "fitness": metrics.get("fitness"),
        },
    )
    log_tracking_table_from_csv(
        tracking_session,
        "tables/training_history",
        config.paths.train_results_csv_path,
        max_rows=config.tracking.max_logged_table_rows,
    )
    log_tracking_images(
        tracking_session,
        build_training_image_mapping(
            run_dir, config.tracking.max_logged_images
        ),
    )
    save_tracking_artifacts(
        tracking_session,
        [
            config.paths.train_metrics_path,
            config.paths.train_results_csv_path,
            config.paths.train_best_weights_path,
            config.paths.train_latest_weights_path,
            config.paths.train_latest_run_path,
            dataset_yaml_path(config.paths.train_dir),
            dataset_manifest_path(config.paths.train_dir),
            dataset_classes_path(config.paths.train_dir),
            run_dir / "metrics.json",
        ],
    )


def build_curriculum_phases(
    curriculum: CurriculumConfig,
    main_epochs_per_stage: int,
) -> list[CurriculumPhase]:
    phases: list[CurriculumPhase] = []
    stages = max(1, curriculum.stages)
    main_epochs = max(0, main_epochs_per_stage)
    for stage_index in range(stages):
        stage_difficulty = stage_index / (stages - 1) if stages > 1 else 0.0
        for phase_name, difficulty, epochs in [
            ("easy", 0.0, 1),
            ("current", stage_difficulty, main_epochs),
            ("hard", 1.0, 1),
        ]:
            if epochs <= 0:
                continue
            train_kwargs = curriculum_train_kwargs(curriculum, difficulty)
            phases.append(
                CurriculumPhase(
                    stage_index=stage_index,
                    phase_name=phase_name,
                    difficulty=difficulty,
                    epochs=epochs,
                    train_kwargs=train_kwargs,
                    augmentations=build_albumentations(curriculum, difficulty),
                )
            )
    return phases


def curriculum_train_kwargs(
    curriculum: CurriculumConfig, difficulty: float
) -> dict[str, float]:
    return {
        "hsv_h": interpolate(curriculum.ranges.hsv_h, difficulty),
        "hsv_s": interpolate(curriculum.ranges.hsv_s, difficulty),
        "hsv_v": interpolate(curriculum.ranges.hsv_v, difficulty),
        "degrees": interpolate(curriculum.ranges.degrees, difficulty),
        "translate": interpolate(curriculum.ranges.translate, difficulty),
        "scale": interpolate(curriculum.ranges.scale, difficulty),
        "shear": interpolate(curriculum.ranges.shear, difficulty),
        "perspective": interpolate(curriculum.ranges.perspective, difficulty),
        "flipud": interpolate(curriculum.ranges.flipud, difficulty),
        "fliplr": interpolate(curriculum.ranges.fliplr, difficulty),
        "bgr": interpolate(curriculum.ranges.bgr, difficulty),
        "mosaic": interpolate(curriculum.ranges.mosaic, difficulty),
        "mixup": interpolate(curriculum.ranges.mixup, difficulty),
        "cutmix": interpolate(curriculum.ranges.cutmix, difficulty),
        "copy_paste": interpolate(curriculum.ranges.copy_paste, difficulty),
    }


def interpolate(value_range: NumericRangeConfig, difficulty: float) -> float:
    difficulty = max(0.0, min(1.0, difficulty))
    return value_range.easy + (value_range.hard - value_range.easy) * difficulty


def build_albumentations(
    curriculum: CurriculumConfig, difficulty: float
) -> list[Any]:
    import albumentations as A

    transforms: list[Any] = []
    for name, transform_config in curriculum.albumentations.items():
        if not transform_config.enabled:
            continue
        values = {
            key: interpolate(value_range, difficulty)
            for key, value_range in transform_config.ranges.items()
        }
        transform = build_albumentation_transform(A, name, values)
        if transform is not None:
            transforms.append(transform)
    return transforms


def build_albumentation_transform(
    albumentations_module: Any, name: str, values: dict[str, float]
) -> Any | None:
    p = values.get("p", 0.0)
    if name == "blur":
        return albumentations_module.Blur(
            blur_limit=odd_limit(values.get("limit", 3)), p=p
        )
    if name == "median_blur":
        return albumentations_module.MedianBlur(
            blur_limit=odd_limit(values.get("limit", 3)), p=p
        )
    if name == "clahe":
        return albumentations_module.CLAHE(
            clip_limit=max(1.0, values.get("clip_limit", 4.0)), p=p
        )
    if name == "random_brightness_contrast":
        return albumentations_module.RandomBrightnessContrast(
            brightness_limit=max(0.0, values.get("brightness_limit", 0.0)),
            contrast_limit=max(0.0, values.get("contrast_limit", 0.0)),
            p=p,
        )
    if name == "image_compression":
        lower = int(round(values.get("quality_lower", 75)))
        upper = int(round(values.get("quality_upper", 100)))
        lower = max(1, min(100, lower))
        upper = max(lower, min(100, upper))
        return albumentations_module.ImageCompression(
            quality_range=(lower, upper), p=p
        )
    return None


def odd_limit(value: float) -> int:
    limit = max(3, int(round(value)))
    return limit if limit % 2 == 1 else limit + 1


def build_training_image_mapping(
    run_dir: Path, max_logged_images: int
) -> dict[str, tuple[Path, str | None]]:
    candidate_images = [
        ("images/training_curves", run_dir / "results.png", "Training curves."),
        (
            "images/confusion_matrix",
            run_dir / "confusion_matrix.png",
            "Validation confusion matrix.",
        ),
        (
            "images/precision_recall_curve",
            run_dir / "PR_curve.png",
            "Precision recall curve.",
        ),
        (
            "images/train_batch_preview",
            run_dir / "train_batch0.jpg",
            "Training batch preview.",
        ),
        (
            "images/val_prediction_preview",
            run_dir / "val_batch0_pred.jpg",
            "Validation prediction preview.",
        ),
    ]

    image_mapping: dict[str, tuple[Path, str | None]] = {}
    for key, image_path, caption in candidate_images:
        if len(image_mapping) >= max_logged_images:
            break
        if image_path.exists():
            image_mapping[key] = (image_path, caption)
    return image_mapping


def resolve_training_device(requested_device: str) -> str:
    normalized_device = requested_device.strip().lower()
    if normalized_device in {"", "cpu"}:
        return "cpu"
    if normalized_device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if normalized_device in {"cuda", "cuda:0", "0"}:
        if torch.cuda.is_available():
            return "0"
        raise RuntimeError(build_cuda_unavailable_message(requested_device))
    if normalized_device.startswith("cuda"):
        if torch.cuda.is_available():
            return normalized_device.replace("cuda:", "")
        raise RuntimeError(build_cuda_unavailable_message(requested_device))
    return requested_device


def build_cuda_unavailable_message(requested_device: str) -> str:
    return (
        f"GPU training was requested with device={requested_device!r}, but this Python environment cannot see CUDA.\n"
        f"torch version: {torch.__version__}\n"
        f"torch.cuda.is_available(): {torch.cuda.is_available()}\n"
        f"torch.cuda.device_count(): {torch.cuda.device_count()}\n"
        "Install a CUDA-enabled PyTorch build in this project's environment, then run training again."
    )


def resolve_training_workers(configured_workers: int) -> int:
    normalized_workers = max(0, int(configured_workers))
    return 0 if sys.platform == "win32" else normalized_workers


def training_device_summary(requested_device: str, selected_device: str) -> str:
    if selected_device == "cpu":
        return f"Training device: cpu (requested={requested_device!r}, torch_cuda_available={torch.cuda.is_available()})"

    device_index = int(selected_device)
    return f"Training device: cuda:{device_index} ({torch.cuda.get_device_name(device_index)})"


def training_worker_summary(
    configured_workers: int, selected_workers: int
) -> str:
    if configured_workers == selected_workers:
        return f"Training data loader workers: {selected_workers}"
    return (
        f"Training data loader workers: {selected_workers} "
        f"(configured={configured_workers}, reduced for Windows stability)"
    )


def copy_output_file(source_path: Path, destination_path: Path) -> None:
    if not non_empty_file(source_path):
        raise FileNotFoundError(
            f"Expected training output not found: {source_path}"
        )
    copy_file(source_path, destination_path)
