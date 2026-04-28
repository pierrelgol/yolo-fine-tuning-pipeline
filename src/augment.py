from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from src.common import (
    TRAIN_SPLIT,
    VAL_SPLIT,
    clear_dir,
    dataset_classes_path,
    dataset_images_dir,
    dataset_labels_dir,
    dataset_manifest_path,
    dataset_predictions_dir,
    discover_images,
    ensure_dir,
    load_class_map,
    ordered_class_names,
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
class SourceImage:
    image_path: Path
    class_id: int
    class_name: str


def augment_with_annotations(
    config: AppConfig,
    background_dir: Path,
    source_image_dir: Path | None = None,
    source_label_dir: Path | None = None,
    classes_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    resolved_background_dir = resolve_path(
        background_dir, base_dir=config.paths.project_root
    )
    resolved_source_dir = resolve_path(
        source_image_dir or Path(config.paths.augment_source_dir),
        base_dir=config.paths.project_root,
    )
    augmented_dataset_dir = resolve_path(
        output_dir or config.paths.augmented_dir,
        base_dir=config.paths.project_root,
    )

    # Discover classes from subfolder names
    class_map, source_images = discover_classes_and_images(resolved_source_dir)
    if not class_map:
        raise FileNotFoundError(
            f"No class subfolders found in {resolved_source_dir}. "
            "Create subfolders like images/<class_name>/ with images inside."
        )
    if not source_images:
        raise FileNotFoundError(
            f"No images found in class subfolders under {resolved_source_dir}"
        )

    background_paths = discover_images(resolved_background_dir)
    if not background_paths:
        raise FileNotFoundError(
            f"No background images found in {resolved_background_dir}"
        )

    scale_min = config.augment.scale_min
    scale_max = config.augment.scale_max
    min_objects = config.augment.min_objects
    max_objects = config.augment.max_objects

    clear_dir(augmented_dataset_dir)
    train_image_dir = dataset_images_dir(augmented_dataset_dir, TRAIN_SPLIT)
    train_label_dir = dataset_labels_dir(augmented_dataset_dir, TRAIN_SPLIT)
    val_image_dir = dataset_images_dir(augmented_dataset_dir, VAL_SPLIT)
    val_label_dir = dataset_labels_dir(augmented_dataset_dir, VAL_SPLIT)
    dataset_predictions_dir(augmented_dataset_dir, TRAIN_SPLIT).mkdir(
        parents=True, exist_ok=True
    )
    dataset_predictions_dir(augmented_dataset_dir, VAL_SPLIT).mkdir(
        parents=True, exist_ok=True
    )

    # Generate one augmented sample per background image
    all_samples = generate_augmented_samples(
        background_paths=background_paths,
        source_images=source_images,
        class_map=class_map,
        scale_min=scale_min,
        scale_max=scale_max,
        min_objects=min_objects,
        max_objects=max_objects,
        seed=config.setup.random_seed,
    )

    train_samples, val_samples = split_items(
        all_samples, config.setup.train_split, config.setup.random_seed
    )

    generated_train = write_augmented_split(
        train_samples, train_image_dir, train_label_dir, config.paths.project_root
    )
    generated_val = write_augmented_split(
        val_samples, val_image_dir, val_label_dir, config.paths.project_root
    )

    save_class_map(dataset_classes_path(augmented_dataset_dir), class_map)
    write_dataset_yaml(augmented_dataset_dir, ordered_class_names(class_map))

    manifest = {
        "dataset_dir": portable_path(
            augmented_dataset_dir, base_dir=config.paths.project_root
        ),
        "background_dir": portable_path(
            resolved_background_dir, base_dir=config.paths.project_root
        ),
        "source_dir": portable_path(
            resolved_source_dir, base_dir=config.paths.project_root
        ),
        "classes": ordered_class_names(class_map),
        "num_train_samples": len(generated_train),
        "num_val_samples": len(generated_val),
        "generated_samples": generated_train + generated_val,
    }
    write_json(dataset_manifest_path(augmented_dataset_dir), manifest)

    print(
        f"Wrote {len(generated_train) + len(generated_val)} augmented samples to {augmented_dataset_dir}"
    )
    return manifest


def discover_classes_and_images(
    source_dir: Path,
) -> tuple[dict[str, int], list[SourceImage]]:
    class_map: dict[str, int] = {}
    source_images: list[SourceImage] = []
    next_id = 0

    if not source_dir.exists():
        return class_map, source_images

    for subfolder in sorted(source_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        class_name = subfolder.name
        images = discover_images(subfolder)
        if not images:
            continue
        class_map[class_name] = next_id
        next_id += 1
        for img_path in images:
            source_images.append(
                SourceImage(
                    image_path=img_path,
                    class_id=class_map[class_name],
                    class_name=class_name,
                )
            )

    return class_map, source_images


@dataclass(frozen=True)
class PlannedObject:
    source: SourceImage
    scale: float
    x: float
    y: float


@dataclass(frozen=True)
class AugmentedSample:
    background_path: Path
    objects: list[PlannedObject]
    sample_name: str


def generate_augmented_samples(
    background_paths: list[Path],
    source_images: list[SourceImage],
    class_map: dict[str, int],
    scale_min: float,
    scale_max: float,
    min_objects: int,
    max_objects: int,
    seed: int,
) -> list[AugmentedSample]:
    rng = random.Random(seed)
    samples: list[AugmentedSample] = []

    # Group source images by class for sampling
    images_by_class: dict[int, list[SourceImage]] = {}
    for src in source_images:
        images_by_class.setdefault(src.class_id, []).append(src)

    for bg_path in background_paths:
        with Image.open(bg_path) as bg_img:
            bg_width, bg_height = bg_img.size

        num_objects = rng.randint(min_objects, max_objects)
        objects: list[PlannedObject] = []

        for _ in range(num_objects):
            src = rng.choice(source_images)
            scale = rng.uniform(scale_min, scale_max)

            # Scale relative to the shorter background dimension
            shorter_bg = min(bg_width, bg_height)
            with Image.open(src.image_path) as src_img:
                src_w, src_h = src_img.size

            scaled_w = int(src_w * scale * shorter_bg / max(src_w, src_h, 1))
            scaled_h = int(src_h * scale * shorter_bg / max(src_w, src_h, 1))
            scaled_w = max(1, scaled_w)
            scaled_h = max(1, scaled_h)

            x = rng.uniform(0, max(0, 1 - scaled_w / bg_width))
            y = rng.uniform(0, max(0, 1 - scaled_h / bg_height))

            objects.append(PlannedObject(source=src, scale=scale, x=x, y=y))

        bg_key = bg_path.as_posix()
        obj_key = ",".join(
            f"{o.source.class_id}:{o.scale:.3f}:{o.x:.3f}:{o.y:.3f}"
            for o in objects
        )
        sample_name = stable_name(bg_key, obj_key, suffix="")

        samples.append(
            AugmentedSample(
                background_path=bg_path,
                objects=objects,
                sample_name=sample_name,
            )
        )

    return samples


def write_augmented_split(
    samples: list[AugmentedSample],
    image_dir: Path,
    label_dir: Path,
    project_root: Path,
) -> list[dict]:
    generated: list[dict] = []
    progress_label = f"augment:{image_dir.parent.name}"

    for sample in tqdm(samples, desc=progress_label, unit="sample"):
        output_img_path, output_label_path = compose_sample(
            sample, image_dir, label_dir
        )
        generated.append(
            {
                "background": portable_path(
                    sample.background_path, base_dir=project_root
                ),
                "output_image": portable_path(
                    output_img_path, base_dir=project_root
                ),
                "output_label": portable_path(
                    output_label_path, base_dir=project_root
                ),
                "num_objects": len(sample.objects),
            }
        )

    return generated


def compose_sample(
    sample: AugmentedSample,
    image_dir: Path,
    label_dir: Path,
) -> tuple[Path, Path]:
    with Image.open(sample.background_path) as bg_img:
        background = bg_img.convert("RGB")

    bg_width, bg_height = background.size
    composite = background.copy()
    label_lines: list[str] = []

    for obj in sample.objects:
        with Image.open(obj.source.image_path) as src_img:
            foreground = src_img.convert("RGB")

        src_w, src_h = foreground.size
        shorter_bg = min(bg_width, bg_height)
        scaled_w = max(
            1, int(src_w * obj.scale * shorter_bg / max(src_w, src_h, 1))
        )
        scaled_h = max(
            1, int(src_h * obj.scale * shorter_bg / max(src_w, src_h, 1))
        )
        foreground = foreground.resize((scaled_w, scaled_h), Image.LANCZOS)

        # Position in pixels
        px = int(obj.x * bg_width)
        py = int(obj.y * bg_height)
        px = max(0, min(px, bg_width - scaled_w))
        py = max(0, min(py, bg_height - scaled_h))

        composite.paste(foreground, (px, py))

        # YOLO label: normalized center x, center y, width, height
        cx = (px + scaled_w / 2) / bg_width
        cy = (py + scaled_h / 2) / bg_height
        w = scaled_w / bg_width
        h = scaled_h / bg_height
        label_lines.append(yolo_label_line(obj.source.class_id, (cx, cy, w, h)))

    output_img_path = image_dir / f"aug_{sample.sample_name}.jpg"
    output_label_path = label_dir / f"aug_{sample.sample_name}.txt"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    composite.save(output_img_path, quality=95)
    save_yolo_labels(output_label_path, label_lines)

    return output_img_path, output_label_path


def write_json(path: Path, payload: dict) -> None:
    import json as _json

    ensure_dir(path.parent)
    path.write_text(
        _json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
