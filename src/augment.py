from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

from PIL import Image

from src.common import (
    copy_image,
    discover_images,
    ensure_dir,
    image_label_path,
    load_class_map,
    parse_yolo_labels,
    save_class_map,
    save_yolo_labels,
    stable_name,
    yolo_label_line,
)
from src.config import AppConfig


@dataclass
class SourceAnnotation:
    image_path: Path
    class_id: int
    class_name: str
    bbox: tuple[float, float, float, float]


def augment_with_annotations(
    config: AppConfig,
    background_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    classes_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    selected_image_dir = image_dir or config.paths.annotation_images_dir
    selected_label_dir = label_dir or config.paths.annotation_labels_dir
    selected_classes_path = classes_path or config.paths.annotation_classes_path
    selected_output_dir = output_dir or config.paths.augmented_dir
    selected_output_classes_path = selected_output_dir / "classes.json"
    selected_output_manifest_path = selected_output_dir / "manifest.json"

    clear_directory(selected_output_dir)
    output_image_dir = ensure_dir(selected_output_dir / "images" / "train2017")
    output_label_dir = ensure_dir(selected_output_dir / "labels" / "train2017")
    output_prediction_dir = ensure_dir(selected_output_dir / "predictions" / "train2017")

    class_map = load_class_map(selected_classes_path)
    annotations = collect_source_annotations(selected_image_dir, selected_label_dir, class_map)
    if not annotations:
        raise FileNotFoundError(f"No annotations found in {selected_label_dir}. Run annotate before augment.")

    background_paths = discover_images(background_dir)
    if not background_paths:
        raise FileNotFoundError(f"No background images found in {background_dir}")

    save_class_map(selected_output_classes_path, class_map)

    generated_samples: list[dict] = []
    for annotation in annotations:
        for background_path in background_paths:
            image_path, label_path = create_augmented_sample(
                annotation=annotation,
                background_path=background_path,
                output_image_dir=output_image_dir,
                output_label_dir=output_label_dir,
            )
            generated_samples.append(
                {
                    "background": str(background_path),
                    "source_image": str(annotation.image_path),
                    "output_image": str(image_path),
                    "output_label": str(label_path),
                    "class_name": annotation.class_name,
                }
            )

    manifest = {
        "image_dir": str(output_image_dir),
        "label_dir": str(output_label_dir),
        "classes_path": str(selected_output_classes_path),
        "background_dir": str(background_dir),
        "prediction_dir": str(output_prediction_dir),
        "num_generated_samples": len(generated_samples),
        "generated_samples": generated_samples,
    }
    selected_output_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(generated_samples)} augmented samples to {selected_output_dir}")
    return manifest


def clear_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    ensure_dir(directory)


def collect_source_annotations(
    image_dir: Path,
    label_dir: Path,
    class_map: dict[str, int],
) -> list[SourceAnnotation]:
    class_name_by_id = {class_id: class_name for class_name, class_id in class_map.items()}
    annotations: list[SourceAnnotation] = []

    for image_path in discover_images(image_dir):
        label_path = image_label_path(image_path, image_dir, label_dir)
        for class_id, bbox in parse_yolo_labels(label_path):
            class_name = class_name_by_id.get(class_id, f"class_{class_id}")
            annotations.append(
                SourceAnnotation(
                    image_path=image_path,
                    class_id=class_id,
                    class_name=class_name,
                    bbox=bbox,
                )
            )

    return annotations


def create_augmented_sample(
    annotation: SourceAnnotation,
    background_path: Path,
    output_image_dir: Path,
    output_label_dir: Path,
) -> tuple[Path, Path]:
    foreground = crop_annotation_from_source_image(annotation)

    with Image.open(background_path) as opened_background:
        background = opened_background.convert("RGBA")

    background_width, background_height = background.size
    resized_width, resized_height = resize_foreground_to_fit_background(foreground, background.size)
    resized_foreground = foreground.resize((resized_width, resized_height))

    x_offset = (background_width - resized_width) // 2
    y_offset = (background_height - resized_height) // 2
    background.alpha_composite(resized_foreground, (x_offset, y_offset))

    bbox_key = ",".join(f"{value:.6f}" for value in annotation.bbox)
    sample_stem = stable_name(
        background_path.as_posix(),
        annotation.image_path.as_posix(),
        str(annotation.class_id),
        bbox_key,
        suffix="",
    )
    output_image_path = output_image_dir / f"augmented_{sample_stem}.jpg"
    output_label_path = output_label_dir / f"augmented_{sample_stem}.txt"

    background.convert("RGB").save(output_image_path, quality=95)
    bbox = (
        (x_offset + resized_width / 2) / background_width,
        (y_offset + resized_height / 2) / background_height,
        resized_width / background_width,
        resized_height / background_height,
    )
    save_yolo_labels(output_label_path, [yolo_label_line(annotation.class_id, bbox)])
    return output_image_path, output_label_path


def crop_annotation_from_source_image(annotation: SourceAnnotation) -> Image.Image:
    with Image.open(annotation.image_path) as opened_source_image:
        source_image = opened_source_image.convert("RGBA")

    left, top, right, bottom = annotation_box_in_pixels(annotation, source_image.size)
    return source_image.crop((left, top, right, bottom))


def annotation_box_in_pixels(annotation: SourceAnnotation, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    image_width, image_height = image_size
    x_center, y_center, width, height = annotation.bbox
    left = max(0, int((x_center - width / 2) * image_width))
    top = max(0, int((y_center - height / 2) * image_height))
    right = min(image_width, int((x_center + width / 2) * image_width))
    bottom = min(image_height, int((y_center + height / 2) * image_height))
    return left, top, right, bottom


def resize_foreground_to_fit_background(foreground: Image.Image, background_size: tuple[int, int]) -> tuple[int, int]:
    background_width, background_height = background_size
    maximum_foreground_width = max(1, background_width // 3)
    maximum_foreground_height = max(1, background_height // 3)

    width_scale = maximum_foreground_width / max(1, foreground.width)
    height_scale = maximum_foreground_height / max(1, foreground.height)
    scale = min(width_scale, height_scale, 1.0)

    resized_width = max(1, int(foreground.width * scale))
    resized_height = max(1, int(foreground.height * scale))
    return resized_width, resized_height
