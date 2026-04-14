from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from src.common import resolve_path as resolve_common_path


@dataclass(frozen=True)
class PathConfig:
    project_root: Path
    image_dir: Path
    dataset_dir: Path
    raw_dir: Path
    coco128_dir: Path
    annotation_dir: Path
    augmented_dir: Path
    train_dir: Path
    train_runs_dir: Path
    train_best_weights_path: Path
    train_latest_weights_path: Path
    train_latest_run_path: Path
    train_metrics_path: Path
    train_results_csv_path: Path
    eval_dir: Path
    eval_runs_dir: Path
    eval_latest_metrics_path: Path
    infer_dir: Path
    infer_runs_dir: Path
    infer_latest_manifest_path: Path
    run_versions_path: Path


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
class WatchConfig:
    source: str
    confidence: float
    image_size: int
    window_name: str


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
    watch: WatchConfig
    tracking: TrackingConfig
    config_path: Path


def load_config(config_path: Path | None = None) -> AppConfig:
    resolved_config_path = resolve_config_path(config_path)
    payload = tomllib.loads(resolved_config_path.read_text(encoding="utf-8"))
    project_root = resolved_config_path.parent.resolve()

    paths_payload = payload.get("paths", {})
    fetch_payload = payload.get("fetch", {})
    setup_payload = payload.get("setup", {})
    train_payload = payload.get("train", {})
    eval_payload = payload.get("eval", {})
    infer_payload = payload.get("infer", {})
    watch_payload = payload.get("watch", {})
    tracking_payload = payload.get("tracking", {})
    hyperparameters_payload = train_payload.get("hyperparameters", {})

    image_dir = resolve_path(project_root, str(paths_payload.get("image_dir", "image")))
    dataset_dir = resolve_path(project_root, str(paths_payload.get("dataset_dir", "dataset")))
    train_dir = dataset_dir / "train"
    eval_dir = dataset_dir / "eval"
    infer_dir = dataset_dir / "infer"

    return AppConfig(
        paths=PathConfig(
            project_root=project_root,
            image_dir=image_dir,
            dataset_dir=dataset_dir,
            raw_dir=dataset_dir / "raw",
            coco128_dir=dataset_dir / "coco128",
            annotation_dir=dataset_dir / "annotation",
            augmented_dir=dataset_dir / "augmented",
            train_dir=train_dir,
            train_runs_dir=train_dir / "runs",
            train_best_weights_path=train_dir / "best.pt",
            train_latest_weights_path=train_dir / "latest.pt",
            train_latest_run_path=train_dir / "latest_run.json",
            train_metrics_path=train_dir / "metrics.json",
            train_results_csv_path=train_dir / "results.csv",
            eval_dir=eval_dir,
            eval_runs_dir=eval_dir / "runs",
            eval_latest_metrics_path=eval_dir / "latest_metrics.json",
            infer_dir=infer_dir,
            infer_runs_dir=infer_dir / "runs",
            infer_latest_manifest_path=infer_dir / "latest_manifest.json",
            run_versions_path=dataset_dir / "run_versions.json",
        ),
        fetch=FetchConfig(
            dataset_url=str(fetch_payload.get("dataset_url", "")),
            archive_name=str(fetch_payload.get("archive_name", "coco128.zip")),
        ),
        setup=SetupConfig(
            train_split=float(setup_payload.get("train_split", 0.8)),
            random_seed=int(setup_payload.get("random_seed", 42)),
        ),
        train=TrainConfig(
            model_name=str(train_payload.get("model_name", "yolo26n.pt")),
            image_size=int(train_payload.get("image_size", 640)),
            epochs=int(train_payload.get("epochs", 10)),
            batch_size=int(train_payload.get("batch_size", 8)),
            device=str(train_payload.get("device", "auto")),
            hyperparameters=TrainHyperparameterConfig(
                patience=int(hyperparameters_payload.get("patience", 100)),
                optimizer=str(hyperparameters_payload.get("optimizer", "auto")),
                initial_learning_rate=float(hyperparameters_payload.get("initial_learning_rate", 0.01)),
                final_learning_rate_factor=float(hyperparameters_payload.get("final_learning_rate_factor", 0.01)),
                momentum=float(hyperparameters_payload.get("momentum", 0.937)),
                weight_decay=float(hyperparameters_payload.get("weight_decay", 0.0005)),
                warmup_epochs=float(hyperparameters_payload.get("warmup_epochs", 3.0)),
                box_loss_gain=float(hyperparameters_payload.get("box_loss_gain", 7.5)),
                class_loss_gain=float(hyperparameters_payload.get("class_loss_gain", 0.5)),
                dfl_loss_gain=float(hyperparameters_payload.get("dfl_loss_gain", 1.5)),
                hue_augmentation=float(hyperparameters_payload.get("hue_augmentation", 0.015)),
                saturation_augmentation=float(hyperparameters_payload.get("saturation_augmentation", 0.7)),
                value_augmentation=float(hyperparameters_payload.get("value_augmentation", 0.4)),
                rotation_degrees=float(hyperparameters_payload.get("rotation_degrees", 0.0)),
                translation_fraction=float(hyperparameters_payload.get("translation_fraction", 0.1)),
                scaling_gain=float(hyperparameters_payload.get("scaling_gain", 0.5)),
                shear_degrees=float(hyperparameters_payload.get("shear_degrees", 0.0)),
                perspective_fraction=float(hyperparameters_payload.get("perspective_fraction", 0.0)),
                vertical_flip_probability=float(hyperparameters_payload.get("vertical_flip_probability", 0.0)),
                horizontal_flip_probability=float(hyperparameters_payload.get("horizontal_flip_probability", 0.5)),
                mosaic_probability=float(hyperparameters_payload.get("mosaic_probability", 1.0)),
                mixup_probability=float(hyperparameters_payload.get("mixup_probability", 0.0)),
                copy_paste_probability=float(hyperparameters_payload.get("copy_paste_probability", 0.0)),
                workers=int(hyperparameters_payload.get("workers", 0)),
            ),
        ),
        evaluate=EvalConfig(
            dataset_yaml=str(eval_payload.get("dataset_yaml", "dataset/train/dataset.yaml")),
        ),
        infer=InferConfig(
            dataset_name=str(infer_payload.get("dataset_name", "augmented")),
        ),
        watch=WatchConfig(
            source=str(watch_payload.get("source", "")).strip(),
            confidence=float(watch_payload.get("confidence", 0.25)),
            image_size=int(watch_payload.get("image_size", 640)),
            window_name=str(watch_payload.get("window_name", "YOLO Watch")),
        ),
        tracking=TrackingConfig(
            enabled=bool(tracking_payload.get("enabled", True)),
            project_name=str(tracking_payload.get("project_name", "yolo-fine-tuning")),
            auto_log_gpu=bool(tracking_payload.get("auto_log_gpu", True)),
            gpu_log_interval_seconds=float(tracking_payload.get("gpu_log_interval_seconds", 10.0)),
            log_every_n_steps=max(1, int(tracking_payload.get("log_every_n_steps", 1))),
            max_logged_images=int(tracking_payload.get("max_logged_images", 6)),
            max_logged_table_rows=int(tracking_payload.get("max_logged_table_rows", 200)),
        ),
        config_path=resolved_config_path,
    )


def resolve_config_path(config_path: Path | None) -> Path:
    if config_path is None:
        resolved_path = Path(__file__).resolve().parent.parent / "config.toml"
    else:
        resolved_path = resolve_common_path(config_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved_path}")

    return resolved_path


def resolve_path(project_root: Path, configured_path: str) -> Path:
    return resolve_path_from_project(project_root, configured_path)


def resolve_path_from_project(project_root: Path, configured_path: str) -> Path:
    return resolve_common_path(configured_path, base_dir=project_root)
