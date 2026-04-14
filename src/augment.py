from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.common import (
    clear_dir,
    dataset_classes_path,
    dataset_images_dir,
    dataset_labels_dir,
    dataset_manifest_path,
    dataset_predictions_dir,
    discover_images,
    image_label_path,
    load_class_map,
    ordered_class_names,
    parse_yolo_labels,
    portable_path,
    resolve_path,
    save_class_map,
    save_yolo_labels,
    split_items,
    stable_name,
    write_dataset_yaml,
    yolo_label_line,
)
from src.config import AppConfig


@dataclass(frozen=True)
class SourceAnnotation:
    image_path: Path
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class PlannedSample:
    annotation: SourceAnnotation
    background_path: Path


def augment_with_annotations(
    config: AppConfig,
    background_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    classes_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    source_dataset_dir = config.paths.annotation_dir
    resolved_background_dir = resolve_path(background_dir, base_dir=config.paths.project_root)
    source_image_dir = resolve_path(image_dir or dataset_images_dir(source_dataset_dir), base_dir=config.paths.project_root)
    source_label_dir = resolve_path(label_dir or dataset_labels_dir(source_dataset_dir), base_dir=config.paths.project_root)
    source_classes_path = resolve_path(classes_path or dataset_classes_path(source_dataset_dir), base_dir=config.paths.project_root)
    augmented_dataset_dir = resolve_path(output_dir or config.paths.augmented_dir, base_dir=config.paths.project_root)

    class_map = load_class_map(source_classes_path)
    if not class_map:
        raise FileNotFoundError(f"No classes found in {source_classes_path}. Run annotate before augment.")

    annotations = collect_source_annotations(source_image_dir, source_label_dir, class_map)
    if not annotations:
        raise FileNotFoundError(f"No annotations found in {source_label_dir}. Run annotate before augment.")

    background_paths = discover_images(resolved_background_dir)
    if not background_paths:
        raise FileNotFoundError(f"No background images found in {resolved_background_dir}")

    clear_dir(augmented_dataset_dir)
    train_image_dir = dataset_images_dir(augmented_dataset_dir, "train2017")
    train_label_dir = dataset_labels_dir(augmented_dataset_dir, "train2017")
    val_image_dir = dataset_images_dir(augmented_dataset_dir, "val2017")
    val_label_dir = dataset_labels_dir(augmented_dataset_dir, "val2017")
    dataset_predictions_dir(augmented_dataset_dir, "train2017").mkdir(parents=True, exist_ok=True)
    dataset_predictions_dir(augmented_dataset_dir, "val2017").mkdir(parents=True, exist_ok=True)

    planned_samples = [PlannedSample(annotation=annotation, background_path=background_path) for annotation in annotations for background_path in background_paths]
    train_samples, val_samples = split_items(planned_samples, config.setup.train_split, config.setup.random_seed)
    train_samples = sorted(train_samples, key=planned_sample_sort_key)
    val_samples = sorted(val_samples, key=planned_sample_sort_key)

    generated_train_samples = write_samples(train_samples, train_image_dir, train_label_dir, config.paths.project_root)
    generated_val_samples = write_samples(val_samples, val_image_dir, val_label_dir, config.paths.project_root)

    save_class_map(dataset_classes_path(augmented_dataset_dir), class_map)
    write_dataset_yaml(augmented_dataset_dir, ordered_class_names(class_map))

    manifest = {
        "dataset_dir": portable_path(augmented_dataset_dir, base_dir=config.paths.project_root),
        "background_dir": portable_path(resolved_background_dir, base_dir=config.paths.project_root),
        "classes_path": portable_path(dataset_classes_path(augmented_dataset_dir), base_dir=config.paths.project_root),
        "num_train_samples": len(generated_train_samples),
        "num_val_samples": len(generated_val_samples),
        "num_generated_samples": len(generated_train_samples) + len(generated_val_samples),
        "generated_samples": generated_train_samples + generated_val_samples,
    }
    dataset_manifest_path(augmented_dataset_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest['num_generated_samples']} augmented samples to {augmented_dataset_dir}")
    return manifest


def collect_source_annotations(image_dir: Path, label_dir: Path, class_map: dict[str, int]) -> list[SourceAnnotation]:
    class_name_by_id = {class_id: class_name for class_name, class_id in class_map.items()}
    annotations: list[SourceAnnotation] = []

    for image_path in discover_images(image_dir):
        label_path = image_label_path(image_path, image_dir, label_dir)
        for class_id, bbox in parse_yolo_labels(label_path):
            annotations.append(
                SourceAnnotation(
                    image_path=image_path,
                    class_id=class_id,
                    class_name=class_name_by_id.get(class_id, f"class_{class_id}"),
                    bbox=bbox,
                )
            )

    return annotations


def planned_sample_sort_key(planned_sample: PlannedSample) -> tuple[str, str, int]:
    return (
        planned_sample.annotation.image_path.name,
        planned_sample.background_path.name,
        planned_sample.annotation.class_id,
    )


def write_samples(
    planned_samples: list[PlannedSample],
    image_dir: Path,
    label_dir: Path,
    project_root: Path,
) -> list[dict]:
    generated_samples: list[dict] = []
    progress_label = f"augment:{image_dir.parent.name}"
    for planned_sample in tqdm(planned_samples, desc=progress_label, unit="sample"):
        output_image_path, output_label_path = create_augmented_sample(
            annotation=planned_sample.annotation,
            background_path=planned_sample.background_path,
            output_image_dir=image_dir,
            output_label_dir=label_dir,
        )
        generated_samples.append(
            {
                "background": portable_path(planned_sample.background_path, base_dir=project_root),
                "source_image": portable_path(planned_sample.annotation.image_path, base_dir=project_root),
                "output_image": portable_path(output_image_path, base_dir=project_root),
                "output_label": portable_path(output_label_path, base_dir=project_root),
                "class_name": planned_sample.annotation.class_name,
            }
        )
    return generated_samples


def create_augmented_sample(
    annotation: SourceAnnotation,
    background_path: Path,
    output_image_dir: Path,
    output_label_dir: Path,
) -> tuple[Path, Path]:
    foreground = crop_foreground(annotation)
    with Image.open(background_path) as opened_background:
        background = opened_background.convert("RGBA")

    background_width, background_height = background.size
    resized_width, resized_height = resize_foreground(foreground.size, background.size)
    resized_foreground = foreground.resize((resized_width, resized_height))

    x_offset = (background_width - resized_width) // 2
    y_offset = (background_height - resized_height) // 2
    background.alpha_composite(resized_foreground, (x_offset, y_offset))

    bbox_key = ",".join(f"{value:.6f}" for value in annotation.bbox)
    sample_name = stable_name(
        background_path.as_posix(),
        annotation.image_path.as_posix(),
        str(annotation.class_id),
        bbox_key,
        suffix="",
    )

    output_image_path = image_dir_path(output_image_dir, sample_name)
    output_label_path = label_dir_path(output_label_dir, sample_name)
    background.convert("RGB").save(output_image_path, quality=95)

    output_bbox = (
        (x_offset + resized_width / 2) / background_width,
        (y_offset + resized_height / 2) / background_height,
        resized_width / background_width,
        resized_height / background_height,
    )
    save_yolo_labels(output_label_path, [yolo_label_line(annotation.class_id, output_bbox)])
    return output_image_path, output_label_path


def crop_foreground(annotation: SourceAnnotation) -> Image.Image:
    with Image.open(annotation.image_path) as opened_source_image:
        source_image = opened_source_image.convert("RGBA")

    image_width, image_height = source_image.size
    x_center, y_center, width, height = annotation.bbox
    left = max(0, int((x_center - width / 2) * image_width))
    top = max(0, int((y_center - height / 2) * image_height))
    right = min(image_width, int((x_center + width / 2) * image_width))
    bottom = min(image_height, int((y_center + height / 2) * image_height))
    return source_image.crop((left, top, right, bottom))


def resize_foreground(foreground_size: tuple[int, int], background_size: tuple[int, int]) -> tuple[int, int]:
    foreground_width, foreground_height = foreground_size
    background_width, background_height = background_size
    max_width = max(1, background_width // 3)
    max_height = max(1, background_height // 3)
    width_scale = max_width / max(1, foreground_width)
    height_scale = max_height / max(1, foreground_height)
    scale = min(width_scale, height_scale, 1.0)
    return max(1, int(foreground_width * scale)), max(1, int(foreground_height * scale))


def image_dir_path(image_dir: Path, sample_name: str) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir / f"augmented_{sample_name}.jpg"


def label_dir_path(label_dir: Path, sample_name: str) -> Path:
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / f"augmented_{sample_name}.txt"
