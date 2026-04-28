from __future__ import annotations

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
    ordered_class_names,
    portable_path,
    resolve_path,
    save_class_map,
    save_yolo_labels,
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
    background_dir: Path | None = None,
    source_image_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    resolved_background_dir = resolve_path(
        background_dir or config.augment.background_dir,
        base_dir=config.paths.project_root,
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
    validate_augment_settings(
        scale_min=scale_min,
        scale_max=scale_max,
        min_objects=min_objects,
        max_objects=max_objects,
    )

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

    train_samples, val_samples = split_augmented_samples(
        all_samples,
        train_split=config.setup.train_split,
        seed=config.setup.random_seed,
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
        "class_counts": count_planned_classes(all_samples, class_map),
        "class_balance": summarize_class_balance(all_samples, class_map),
        "train_class_balance": summarize_class_balance(
            train_samples, class_map
        ),
        "val_class_balance": summarize_class_balance(val_samples, class_map),
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

    images_by_class: dict[int, list[SourceImage]] = {}
    for src in source_images:
        images_by_class.setdefault(src.class_id, []).append(src)
    class_ids = sorted(images_by_class)
    class_counts = dict.fromkeys(class_ids, 0)
    object_counts = plan_object_counts(
        num_backgrounds=len(background_paths),
        num_classes=len(class_ids),
        min_objects=min_objects,
        max_objects=max_objects,
        rng=rng,
    )

    for bg_path, num_objects in zip(background_paths, object_counts, strict=True):
        with Image.open(bg_path) as bg_img:
            bg_width, bg_height = bg_img.size

        num_objects = min(num_objects, len(class_ids))
        objects: list[PlannedObject] = []

        for class_id in choose_balanced_classes(
            class_ids, class_counts, num_objects, rng
        ):
            src = rng.choice(images_by_class[class_id])
            class_counts[class_id] += 1
            scale = rng.uniform(scale_min, scale_max)

            with Image.open(src.image_path) as src_img:
                src_w, src_h = src_img.size

            scaled_w, scaled_h = scaled_foreground_size(
                source_size=(src_w, src_h),
                background_size=(bg_width, bg_height),
                scale=scale,
            )

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


def validate_augment_settings(
    scale_min: float, scale_max: float, min_objects: int, max_objects: int
) -> None:
    if scale_min <= 0 or scale_max <= 0:
        raise ValueError("augment scale_min and scale_max must be positive")
    if scale_min > scale_max:
        raise ValueError("augment scale_min cannot be greater than scale_max")
    if min_objects < 1:
        raise ValueError("augment min_objects must be at least 1")
    if max_objects < min_objects:
        raise ValueError("augment max_objects cannot be less than min_objects")


def plan_object_counts(
    num_backgrounds: int,
    num_classes: int,
    min_objects: int,
    max_objects: int,
    rng: random.Random,
) -> list[int]:
    max_total_objects = num_backgrounds * max_objects
    if max_total_objects < num_classes:
        raise ValueError(
            "Not enough object slots to represent every class. "
            f"Need at least {num_classes} slots, but {num_backgrounds} "
            f"backgrounds * max_objects={max_objects} gives {max_total_objects}."
        )

    object_counts = [
        rng.randint(min_objects, max_objects) for _ in range(num_backgrounds)
    ]
    while sum(object_counts) < num_classes:
        expandable_indices = [
            index
            for index, count in enumerate(object_counts)
            if count < max_objects
        ]
        if not expandable_indices:
            break
        index = rng.choice(expandable_indices)
        object_counts[index] += 1
    return object_counts


def choose_balanced_classes(
    class_ids: list[int],
    class_counts: dict[int, int],
    num_objects: int,
    rng: random.Random,
) -> list[int]:
    ranked_class_ids = sorted(
        class_ids, key=lambda class_id: (class_counts[class_id], rng.random())
    )
    return ranked_class_ids[:num_objects]


def count_planned_classes(
    samples: list[AugmentedSample], class_map: dict[str, int]
) -> dict[str, int]:
    id_to_name = {class_id: name for name, class_id in class_map.items()}
    counts = {name: 0 for name in ordered_class_names(class_map)}
    for sample in samples:
        for obj in sample.objects:
            counts[id_to_name[obj.source.class_id]] += 1
    return counts


def summarize_class_balance(
    samples: list[AugmentedSample], class_map: dict[str, int]
) -> dict[str, int | bool]:
    counts = count_planned_classes(samples, class_map)
    values = list(counts.values())
    minimum = min(values) if values else 0
    maximum = max(values) if values else 0
    return {
        "min_count": minimum,
        "max_count": maximum,
        "spread": maximum - minimum,
        "all_classes_present": all(value > 0 for value in values),
    }


def split_augmented_samples(
    samples: list[AugmentedSample],
    train_split: float,
    seed: int,
) -> tuple[list[AugmentedSample], list[AugmentedSample]]:
    ordered_samples = list(samples)
    if len(ordered_samples) <= 1:
        return ordered_samples, []

    rng = random.Random(seed)
    rng.shuffle(ordered_samples)
    train_count = int(len(ordered_samples) * train_split)
    train_count = max(1, min(len(ordered_samples) - 1, train_count))
    val_count = len(ordered_samples) - train_count

    selected_val_indices: set[int] = set()
    covered_classes: set[int] = set()
    all_classes = {
        obj.source.class_id
        for sample in ordered_samples
        for obj in sample.objects
    }

    while len(selected_val_indices) < val_count and covered_classes != all_classes:
        best_index: int | None = None
        best_new_coverage: set[int] = set()
        for index, sample in enumerate(ordered_samples):
            if index in selected_val_indices:
                continue
            sample_classes = {obj.source.class_id for obj in sample.objects}
            new_coverage = sample_classes - covered_classes
            if len(new_coverage) > len(best_new_coverage):
                best_index = index
                best_new_coverage = new_coverage
        if best_index is None or not best_new_coverage:
            break
        selected_val_indices.add(best_index)
        covered_classes.update(best_new_coverage)

    remaining_indices = [
        index
        for index in range(len(ordered_samples))
        if index not in selected_val_indices
    ]
    rng.shuffle(remaining_indices)
    selected_val_indices.update(
        remaining_indices[: max(0, val_count - len(selected_val_indices))]
    )

    val_samples = [
        sample
        for index, sample in enumerate(ordered_samples)
        if index in selected_val_indices
    ]
    train_samples = [
        sample
        for index, sample in enumerate(ordered_samples)
        if index not in selected_val_indices
    ]
    return train_samples, val_samples


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
            foreground = src_img.convert("RGBA")

        scaled_w, scaled_h = scaled_foreground_size(
            source_size=foreground.size,
            background_size=(bg_width, bg_height),
            scale=obj.scale,
        )
        foreground = foreground.resize((scaled_w, scaled_h), Image.LANCZOS)

        px = int(obj.x * bg_width)
        py = int(obj.y * bg_height)
        px = max(0, min(px, max(0, bg_width - scaled_w)))
        py = max(0, min(py, max(0, bg_height - scaled_h)))
        left, top, right, bottom = clipped_box(
            px, py, scaled_w, scaled_h, bg_width, bg_height
        )
        if right <= left or bottom <= top:
            continue

        crop = foreground.crop((left - px, top - py, right - px, bottom - py))
        composite.paste(crop.convert("RGB"), (left, top), crop.getchannel("A"))

        cx = ((left + right) / 2) / bg_width
        cy = ((top + bottom) / 2) / bg_height
        w = (right - left) / bg_width
        h = (bottom - top) / bg_height
        label_lines.append(yolo_label_line(obj.source.class_id, (cx, cy, w, h)))

    output_img_path = image_dir / f"aug_{sample.sample_name}.jpg"
    output_label_path = label_dir / f"aug_{sample.sample_name}.txt"

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    composite.save(output_img_path, quality=95)
    save_yolo_labels(output_label_path, label_lines)

    return output_img_path, output_label_path


def scaled_foreground_size(
    source_size: tuple[int, int],
    background_size: tuple[int, int],
    scale: float,
) -> tuple[int, int]:
    src_w, src_h = source_size
    bg_width, bg_height = background_size
    shorter_bg = min(bg_width, bg_height)
    source_long_edge = max(src_w, src_h, 1)
    scaled_w = max(1, int(src_w * scale * shorter_bg / source_long_edge))
    scaled_h = max(1, int(src_h * scale * shorter_bg / source_long_edge))
    return min(scaled_w, bg_width), min(scaled_h, bg_height)


def clipped_box(
    px: int,
    py: int,
    width: int,
    height: int,
    bg_width: int,
    bg_height: int,
) -> tuple[int, int, int, int]:
    left = max(0, px)
    top = max(0, py)
    right = min(bg_width, px + width)
    bottom = min(bg_height, py + height)
    return left, top, right, bottom


def write_json(path: Path, payload: dict) -> None:
    import json as _json

    ensure_dir(path.parent)
    path.write_text(
        _json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
