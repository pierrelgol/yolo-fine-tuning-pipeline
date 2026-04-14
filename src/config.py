from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    source_dir: Path
    image_dir: Path
    dataset_dir: Path
    raw_dir: Path
    run_versions_path: Path
    coco128_dir: Path
    annotation_dir: Path
    augmented_dir: Path
    train_dir: Path
    eval_dir: Path
    infer_dir: Path
    coco128_images_dir: Path
    coco128_labels_dir: Path
    coco128_dataset_yaml_path: Path
    annotation_images_dir: Path
    annotation_labels_dir: Path
    annotation_classes_path: Path
    annotation_manifest_path: Path
    augmented_images_dir: Path
    augmented_labels_dir: Path
    augmented_train_images_dir: Path
    augmented_train_labels_dir: Path
    augmented_val_images_dir: Path
    augmented_val_labels_dir: Path
    augmented_classes_path: Path
    augmented_manifest_path: Path
    train_data_dir: Path
    train_images_dir: Path
    train_labels_dir: Path
    train_train_images_dir: Path
    train_train_labels_dir: Path
    train_val_images_dir: Path
    train_val_labels_dir: Path
    train_dataset_yaml_path: Path
    train_manifest_path: Path
    train_metrics_path: Path
    train_results_csv_path: Path
    train_best_weights_path: Path
    train_latest_weights_path: Path
    train_latest_run_path: Path
    train_runs_dir: Path
    eval_latest_metrics_path: Path
    infer_latest_manifest_path: Path


@dataclass(frozen=True)
class FetchConfig:
    dataset_url: str
    archive_name: str


@dataclass(frozen=True)
class SetupConfig:
    train_split: float
    random_seed: int


@dataclass(frozen=True)
class TrainHyperparameterConfig:
    patience: int
    optimizer: str
    initial_learning_rate: float
    final_learning_rate_factor: float
    momentum: float
    weight_decay: float
    warmup_epochs: float
    box_loss_gain: float
    class_loss_gain: float
    dfl_loss_gain: float
    hue_augmentation: float
    saturation_augmentation: float
    value_augmentation: float
    rotation_degrees: float
    translation_fraction: float
    scaling_gain: float
    shear_degrees: float
    perspective_fraction: float
    vertical_flip_probability: float
    horizontal_flip_probability: float
    mosaic_probability: float
    mixup_probability: float
    copy_paste_probability: float
    workers: int


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    image_size: int
    epochs: int
    batch_size: int
    device: str
    hyperparameters: TrainHyperparameterConfig


@dataclass(frozen=True)
class EvalConfig:
    dataset_yaml: str


@dataclass(frozen=True)
class InferConfig:
    dataset_name: str


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    project_name: str
    auto_log_gpu: bool
    gpu_log_interval_seconds: float
    log_every_n_steps: int
    max_logged_images: int
    max_logged_table_rows: int


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig
    fetch: FetchConfig
    setup: SetupConfig
    train: TrainConfig
    evaluate: EvalConfig
    infer: InferConfig
    tracking: TrackingConfig
    config_path: Path


def load_config(config_path: Path | None = None) -> AppConfig:
    resolved_config_path = resolve_config_path(config_path)
    config_payload = tomllib.loads(resolved_config_path.read_text(encoding="utf-8"))
    project_root = resolved_config_path.parent.resolve()

    paths_payload = config_payload.get("paths", {})
    fetch_payload = config_payload.get("fetch", {})
    setup_payload = config_payload.get("setup", {})
    train_payload = config_payload.get("train", {})
    train_hyperparameters_payload = train_payload.get("hyperparameters", {})
    eval_payload = config_payload.get("eval", {})
    infer_payload = config_payload.get("infer", {})
    tracking_payload = config_payload.get("tracking", {})

    image_dir = resolve_path(project_root, paths_payload.get("image_dir", "image"))
    dataset_dir = resolve_path(project_root, paths_payload.get("dataset_dir", "dataset"))
    source_dir = project_root / "src"

    raw_dir = dataset_dir / "raw"
    coco128_dir = dataset_dir / "coco128"
    annotation_dir = dataset_dir / "annotation"
    augmented_dir = dataset_dir / "augmented"
    train_dir = dataset_dir / "train"
    eval_dir = dataset_dir / "eval"
    infer_dir = dataset_dir / "infer"

    train_data_dir = train_dir / "data"
    train_images_dir = train_data_dir / "images"
    train_labels_dir = train_data_dir / "labels"

    path_config = PathConfig(
        project_root=project_root,
        source_dir=source_dir,
        image_dir=image_dir,
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        run_versions_path=dataset_dir / "run_versions.json",
        coco128_dir=coco128_dir,
        annotation_dir=annotation_dir,
        augmented_dir=augmented_dir,
        train_dir=train_dir,
        eval_dir=eval_dir,
        infer_dir=infer_dir,
        coco128_images_dir=coco128_dir / "images" / "train2017",
        coco128_labels_dir=coco128_dir / "labels" / "train2017",
        coco128_dataset_yaml_path=coco128_dir / "dataset.yaml",
        annotation_images_dir=annotation_dir / "images" / "train2017",
        annotation_labels_dir=annotation_dir / "labels" / "train2017",
        annotation_classes_path=annotation_dir / "classes.json",
        annotation_manifest_path=annotation_dir / "manifest.json",
        augmented_images_dir=augmented_dir / "images" / "train2017",
        augmented_labels_dir=augmented_dir / "labels" / "train2017",
        augmented_train_images_dir=augmented_dir / "images" / "train2017",
        augmented_train_labels_dir=augmented_dir / "labels" / "train2017",
        augmented_val_images_dir=augmented_dir / "images" / "val2017",
        augmented_val_labels_dir=augmented_dir / "labels" / "val2017",
        augmented_classes_path=augmented_dir / "classes.json",
        augmented_manifest_path=augmented_dir / "manifest.json",
        train_data_dir=train_data_dir,
        train_images_dir=train_images_dir,
        train_labels_dir=train_labels_dir,
        train_train_images_dir=train_images_dir / "train2017",
        train_train_labels_dir=train_labels_dir / "train2017",
        train_val_images_dir=train_images_dir / "val2017",
        train_val_labels_dir=train_labels_dir / "val2017",
        train_dataset_yaml_path=train_dir / "dataset.yaml",
        train_manifest_path=train_dir / "manifest.json",
        train_metrics_path=train_dir / "metrics.json",
        train_results_csv_path=train_dir / "results.csv",
        train_best_weights_path=train_dir / "best.pt",
        train_latest_weights_path=train_dir / "latest.pt",
        train_latest_run_path=train_dir / "latest_run.json",
        train_runs_dir=train_dir,
        eval_latest_metrics_path=eval_dir / "latest_metrics.json",
        infer_latest_manifest_path=infer_dir / "latest_manifest.json",
    )

    fetch_config = FetchConfig(
        dataset_url=str(fetch_payload.get("dataset_url", "")),
        archive_name=str(fetch_payload.get("archive_name", "coco128.zip")),
    )
    setup_config = SetupConfig(
        train_split=float(setup_payload.get("train_split", 0.8)),
        random_seed=int(setup_payload.get("random_seed", 42)),
    )
    train_config = TrainConfig(
        model_name=str(train_payload.get("model_name", "yolo26n.pt")),
        image_size=int(train_payload.get("image_size", 640)),
        epochs=int(train_payload.get("epochs", 10)),
        batch_size=int(train_payload.get("batch_size", 8)),
        device=str(train_payload.get("device", "auto")),
        hyperparameters=TrainHyperparameterConfig(
            patience=int(train_hyperparameters_payload.get("patience", 100)),
            optimizer=str(train_hyperparameters_payload.get("optimizer", "auto")),
            initial_learning_rate=float(train_hyperparameters_payload.get("initial_learning_rate", 0.01)),
            final_learning_rate_factor=float(train_hyperparameters_payload.get("final_learning_rate_factor", 0.01)),
            momentum=float(train_hyperparameters_payload.get("momentum", 0.937)),
            weight_decay=float(train_hyperparameters_payload.get("weight_decay", 0.0005)),
            warmup_epochs=float(train_hyperparameters_payload.get("warmup_epochs", 3.0)),
            box_loss_gain=float(train_hyperparameters_payload.get("box_loss_gain", 7.5)),
            class_loss_gain=float(train_hyperparameters_payload.get("class_loss_gain", 0.5)),
            dfl_loss_gain=float(train_hyperparameters_payload.get("dfl_loss_gain", 1.5)),
            hue_augmentation=float(train_hyperparameters_payload.get("hue_augmentation", 0.015)),
            saturation_augmentation=float(train_hyperparameters_payload.get("saturation_augmentation", 0.7)),
            value_augmentation=float(train_hyperparameters_payload.get("value_augmentation", 0.4)),
            rotation_degrees=float(train_hyperparameters_payload.get("rotation_degrees", 0.0)),
            translation_fraction=float(train_hyperparameters_payload.get("translation_fraction", 0.1)),
            scaling_gain=float(train_hyperparameters_payload.get("scaling_gain", 0.5)),
            shear_degrees=float(train_hyperparameters_payload.get("shear_degrees", 0.0)),
            perspective_fraction=float(train_hyperparameters_payload.get("perspective_fraction", 0.0)),
            vertical_flip_probability=float(train_hyperparameters_payload.get("vertical_flip_probability", 0.0)),
            horizontal_flip_probability=float(train_hyperparameters_payload.get("horizontal_flip_probability", 0.5)),
            mosaic_probability=float(train_hyperparameters_payload.get("mosaic_probability", 1.0)),
            mixup_probability=float(train_hyperparameters_payload.get("mixup_probability", 0.0)),
            copy_paste_probability=float(train_hyperparameters_payload.get("copy_paste_probability", 0.0)),
            workers=int(train_hyperparameters_payload.get("workers", 0)),
        ),
    )
    eval_config = EvalConfig(dataset_yaml=str(eval_payload.get("dataset_yaml", "dataset/train/dataset.yaml")))
    infer_config = InferConfig(dataset_name=str(infer_payload.get("dataset_name", "augmented")))
    tracking_config = TrackingConfig(
        enabled=bool(tracking_payload.get("enabled", True)),
        project_name=str(tracking_payload.get("project_name", "yolo-fine-tuning-pipeline")),
        auto_log_gpu=bool(tracking_payload.get("auto_log_gpu", True)),
        gpu_log_interval_seconds=float(tracking_payload.get("gpu_log_interval_seconds", 10.0)),
        log_every_n_steps=max(1, int(tracking_payload.get("log_every_n_steps", 10))),
        max_logged_images=int(tracking_payload.get("max_logged_images", 6)),
        max_logged_table_rows=int(tracking_payload.get("max_logged_table_rows", 200)),
    )

    return AppConfig(
        paths=path_config,
        fetch=fetch_config,
        setup=setup_config,
        train=train_config,
        evaluate=eval_config,
        infer=infer_config,
        tracking=tracking_config,
        config_path=resolved_config_path,
    )


def resolve_config_path(config_path: Path | None) -> Path:
    if config_path is None:
        resolved_path = Path(__file__).resolve().parent.parent / "config.toml"
    else:
        resolved_path = Path(config_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    return resolved_path


def resolve_path(project_root: Path, configured_path: str) -> Path:
    candidate_path = Path(configured_path)
    if candidate_path.is_absolute():
        return candidate_path
    return (project_root / candidate_path).resolve()
