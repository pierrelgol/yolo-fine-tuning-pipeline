from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
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


@dataclass(frozen=False)
class PlannedSample:
    annotation: SourceAnnotation
    background_path: Path
    complexity_offset: float = 0.0


PHI = (1 + math.sqrt(5)) / 2


def augment_with_annotations(
    config: AppConfig,
    background_dir: Path,
    image_dir: Path | None = None,
    label_dir: Path | None = None,
    classes_path: Path | None = None,
    output_dir: Path | None = None,
    curriculum_complexity: float = 0.0,
) -> dict:
    source_dataset_dir = config.paths.annotation_dir
    resolved_background_dir = resolve_path(
        background_dir, base_dir=config.paths.project_root
    )
    source_image_dir = resolve_path(
        image_dir or dataset_images_dir(source_dataset_dir),
        base_dir=config.paths.project_root,
    )
    source_label_dir = resolve_path(
        label_dir or dataset_labels_dir(source_dataset_dir),
        base_dir=config.paths.project_root,
    )
    source_classes_path = resolve_path(
        classes_path or dataset_classes_path(source_dataset_dir),
        base_dir=config.paths.project_root,
    )
    augmented_dataset_dir = resolve_path(
        output_dir or config.paths.augmented_dir,
        base_dir=config.paths.project_root,
    )

    class_map = load_class_map(source_classes_path)
    if not class_map:
        raise FileNotFoundError(
            f"No classes found in {source_classes_path}. Run annotate before augment."
        )

    annotations = collect_source_annotations(
        source_image_dir, source_label_dir, class_map
    )
    if not annotations:
        raise FileNotFoundError(
            f"No annotations found in {source_label_dir}. Run annotate before augment."
        )

    background_paths = discover_images(resolved_background_dir)
    if not background_paths:
        raise FileNotFoundError(
            f"No background images found in {resolved_background_dir}"
        )

    clear_dir(augmented_dataset_dir)
    train_image_dir = dataset_images_dir(augmented_dataset_dir, "train2017")
    train_label_dir = dataset_labels_dir(augmented_dataset_dir, "train2017")
    val_image_dir = dataset_images_dir(augmented_dataset_dir, "val2017")
    val_label_dir = dataset_labels_dir(augmented_dataset_dir, "val2017")
    dataset_predictions_dir(augmented_dataset_dir, "train2017").mkdir(
        parents=True, exist_ok=True
    )
    dataset_predictions_dir(augmented_dataset_dir, "val2017").mkdir(
        parents=True, exist_ok=True
    )

    planned_samples = [
        PlannedSample(annotation=annotation, background_path=background_path)
        for annotation in annotations
        for background_path in background_paths
    ]
    train_samples, val_samples = split_items(
        planned_samples, config.setup.train_split, config.setup.random_seed
    )

    # Apply curriculum complexity: higher complexity for later samples
    if len(train_samples) > 0:
        total_train = len(train_samples)
        for i, sample in enumerate(train_samples):
            # Complexity increases from 0.0 to 1.0 across the training set
            sample.complexity_offset = (i / total_train) * curriculum_complexity

    train_samples = sorted(train_samples, key=planned_sample_sort_key)
    val_samples = sorted(val_samples, key=planned_sample_sort_key)

    generated_train_samples = write_samples(
        train_samples,
        train_image_dir,
        train_label_dir,
        config.paths.project_root,
    )
    generated_val_samples = write_samples(
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
        "classes_path": portable_path(
            dataset_classes_path(augmented_dataset_dir),
            base_dir=config.paths.project_root,
        ),
        "num_train_samples": len(generated_train_samples),
        "num_val_samples": len(generated_val_samples),
        "num_generated_samples": len(generated_train_samples)
        + len(generated_val_samples),
        "generated_samples": generated_train_samples + generated_val_samples,
    }
    dataset_manifest_path(augmented_dataset_dir).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {manifest['num_generated_samples']} augmented samples to {augmented_dataset_dir}"
    )
    return manifest


def collect_source_annotations(
    image_dir: Path, label_dir: Path, class_map: dict[str, int]
) -> list[SourceAnnotation]:
    class_name_by_id = {
        class_id: class_name for class_name, class_id in class_map.items()
    }
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


def planned_sample_sort_key(
    planned_sample: PlannedSample,
) -> tuple[str, str, int]:
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
            complexity_offset=planned_sample.complexity_offset,
        )
        generated_samples.append(
            {
                "background": portable_path(
                    planned_sample.background_path, base_dir=project_root
                ),
                "source_image": portable_path(
                    planned_sample.annotation.image_path, base_dir=project_root
                ),
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
    complexity_offset: float = 0.0,
) -> tuple[Path, Path]:
    bbox_key = ",".join(f"{value:.6f}" for value in annotation.bbox)
    sample_name = stable_name(
        background_path.as_posix(),
        annotation.image_path.as_posix(),
        str(annotation.class_id),
        bbox_key,
        suffix="",
    )
    seed = int(sample_name, 16)

    foreground = crop_foreground(annotation)
    with Image.open(background_path) as opened_background:
        background = opened_background.convert("RGB")

    composite_image, output_bbox = compose_foreground_with_context(
        foreground=foreground,
        background=background,
        class_id=annotation.class_id,
        seed=seed,
        complexity_offset=complexity_offset,
    )

    output_image_path = image_dir_path(output_image_dir, sample_name)
    output_label_path = label_dir_path(output_label_dir, sample_name)
    composite_image.save(output_image_path, quality=95)

    save_yolo_labels(
        output_label_path, [yolo_label_line(annotation.class_id, output_bbox)]
    )
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


def resize_foreground(
    foreground_size: tuple[int, int], background_size: tuple[int, int]
) -> tuple[int, int]:
    foreground_width, foreground_height = foreground_size
    background_width, background_height = background_size
    max_width = max(1, background_width // 3)
    max_height = max(1, background_height // 3)
    width_scale = max_width / max(1, foreground_width)
    height_scale = max_height / max(1, foreground_height)
    scale = min(width_scale, height_scale, 1.0)
    return max(1, int(foreground_width * scale)), max(1, int(foreground_height * scale))


def compose_foreground_with_context(
    foreground: Image.Image,
    background: Image.Image,
    class_id: int,
    seed: int,
    complexity_offset: float = 0.0,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    foreground_rgba = np.array(foreground.convert("RGBA"))
    background_rgb = cv2.cvtColor(
        np.array(background.convert("RGB")), cv2.COLOR_RGB2BGR
    )

    resized_foreground = resize_foreground_for_context(
        foreground_rgba, background_rgb.shape[:2], np_rng
    )

    warped_rgb, warped_alpha = warp_foreground(
        resized_foreground,
        class_id=class_id,
        rng=rng,
        complexity_offset=complexity_offset,
    )
    composite_bgr, output_bbox = place_warped_foreground(
        background_rgb, warped_rgb, warped_alpha, rng=rng
    )

    composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(composite_rgb), output_bbox


def resize_foreground_for_context(
    foreground_rgba: np.ndarray,
    background_shape: tuple[int, int],
    np_rng: np.random.Generator,
) -> np.ndarray:
    foreground_height, foreground_width = foreground_rgba.shape[:2]
    background_height, background_width = background_shape
    foreground_area = max(1, foreground_height * foreground_width)
    background_area = max(1, background_height * background_width)

    placement_roll = np_rng.random()
    if placement_roll < 0.8:
        target_ratio = np_rng.uniform(0.25, 0.30)
    elif placement_roll < 0.9:
        target_ratio = np_rng.uniform(0.08, 0.095)
    else:
        target_ratio = np_rng.uniform(0.40, 0.50)

    scale = math.sqrt((target_ratio * background_area) / foreground_area)
    resized_width = max(1, int(round(foreground_width * scale)))
    resized_height = max(1, int(round(foreground_height * scale)))
    return cv2.resize(
        foreground_rgba,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def warp_foreground(
    foreground_rgba: np.ndarray,
    class_id: int,
    rng: random.Random,
    complexity_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(foreground_rgba[:, :, :3], cv2.COLOR_RGBA2BGR)
    alpha = foreground_rgba[:, :, 3]
    height, width = alpha.shape

    yaw_deg, pitch_deg = choose_yaw_and_pitch_deg(class_id, rng)
    # Amplify transformations based on complexity_offset
    yaw_deg *= 1.0 + complexity_offset
    pitch_deg *= 1.0 + complexity_offset

    homography = homography_from_yaw_pitch(
        width, height, yaw_deg=yaw_deg, pitch_deg=pitch_deg, f=1000.0
    )
    warped_rgb, translated_homography = warp_perspective_full(rgb, homography)
    warped_alpha = cv2.warpPerspective(
        alpha,
        translated_homography,
        (warped_rgb.shape[1], warped_rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_rgb, warped_alpha


def place_warped_foreground(
    background_bgr: np.ndarray,
    warped_rgb: np.ndarray,
    warped_alpha: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    background_height, background_width = background_bgr.shape[:2]
    warped_rgb, warped_alpha = fit_warped_foreground_to_background(
        warped_rgb,
        warped_alpha,
        background_size=(background_width, background_height),
    )
    warped_height, warped_width = warped_rgb.shape[:2]
    if warped_height == 0 or warped_width == 0:
        raise ValueError("Warped foreground is empty")

    max_x = max(0, background_width - warped_width)
    max_y = max(0, background_height - warped_height)
    x_offset = rng.randint(0, max_x) if max_x > 0 else 0
    y_offset = rng.randint(0, max_y) if max_y > 0 else 0

    alpha_mask = warped_alpha.astype(np.float32) / 255.0
    alpha_mask = alpha_mask[:, :, None]

    composite = background_bgr.copy()
    target_roi = composite[
        y_offset : y_offset + warped_height,
        x_offset : x_offset + warped_width,
    ]
    blended_roi = (
        target_roi.astype(np.float32) * (1.0 - alpha_mask)
        + warped_rgb.astype(np.float32) * alpha_mask
    )
    composite[
        y_offset : y_offset + warped_height,
        x_offset : x_offset + warped_width,
    ] = np.clip(blended_roi, 0, 255).astype(np.uint8)

    output_bbox = bbox_from_alpha(
        warped_alpha,
        x_offset=x_offset,
        y_offset=y_offset,
        background_size=(background_width, background_height),
    )
    return composite, output_bbox


def fit_warped_foreground_to_background(
    warped_rgb: np.ndarray,
    warped_alpha: np.ndarray,
    background_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    background_width, background_height = background_size
    warped_height, warped_width = warped_rgb.shape[:2]

    if warped_width <= background_width and warped_height <= background_height:
        return warped_rgb, warped_alpha

    width_scale = background_width / max(1, warped_width)
    height_scale = background_height / max(1, warped_height)
    scale = min(width_scale, height_scale, 1.0)
    resized_width = max(1, int(math.floor(warped_width * scale)))
    resized_height = max(1, int(math.floor(warped_height * scale)))

    resized_rgb = cv2.resize(
        warped_rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    resized_alpha = cv2.resize(
        warped_alpha,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized_rgb, resized_alpha


def bbox_from_alpha(
    alpha: np.ndarray,
    x_offset: int,
    y_offset: int,
    background_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    background_width, background_height = background_size
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Foreground alpha mask is empty after warping")

    x_min = int(xs.min()) + x_offset
    y_min = int(ys.min()) + y_offset
    x_max = int(xs.max()) + x_offset
    y_max = int(ys.max()) + y_offset

    width = max(1, x_max - x_min + 1)
    height = max(1, y_max - y_min + 1)
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return (
        x_center / background_width,
        y_center / background_height,
        width / background_width,
        height / background_height,
    )


def choose_yaw_and_pitch_deg(class_id: int, rng: random.Random) -> tuple[float, float]:
    probability = rng.random()

    if class_id == 0:
        if probability < 0.1:
            yaw_deg = rng.uniform(-5.0, 5.0)
            pitch_deg = rng.uniform(-3.0, 3.0)
        elif probability < 0.35:
            yaw_deg = rng.uniform(35.0, 55.0)
            pitch_deg = rng.uniform(-15.0, 15.0)
        else:
            yaw_deg = rng.uniform(-55.0, 35.0)
            pitch_deg = rng.uniform(-15.0, 15.0)
        return yaw_deg, pitch_deg

    if class_id == 1:
        if probability < 0.5:
            pitch_deg = rng.uniform(-80.0, -65.0)
            yaw_deg = rng.uniform(-8.0, 8.0)
        else:
            pitch_deg = rng.uniform(-75.0, -60.0)
            yaw_deg = rng.uniform(-20.0, 20.0)
            yaw_deg *= pitch_deg / 70.0
        return yaw_deg, pitch_deg

    yaw_range = 12.0 / PHI
    pitch_range = 10.0 / PHI
    return (
        rng.uniform(-yaw_range, yaw_range),
        rng.uniform(-pitch_range, pitch_range),
    )


def homography_from_yaw_pitch(
    width: int,
    height: int,
    yaw_deg: float,
    pitch_deg: float,
    f: float,
) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    rotation_yaw = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rotation_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    rotation = rotation_pitch @ rotation_yaw

    cx = width / 2.0
    cy = height / 2.0
    source_points = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float32,
    )
    plane_points = np.array(
        [[-cx, -cy, 0.0], [cx, -cy, 0.0], [cx, cy, 0.0], [-cx, cy, 0.0]],
        dtype=np.float32,
    )

    projected_points: list[list[float]] = []
    for point in plane_points:
        rotated = rotation @ point
        z_depth = f + rotated[2]
        projected_points.append(
            [
                float(f * rotated[0] / z_depth + cx),
                float(f * rotated[1] / z_depth + cy),
            ]
        )

    return cv2.getPerspectiveTransform(
        source_points, np.array(projected_points, dtype=np.float32)
    )


def warp_perspective_full(
    image: np.ndarray, homography: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    corners = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        dtype=np.float32,
    )
    warped_corners = cv2.perspectiveTransform(corners[None, :, :], homography)[0]

    x_min, y_min = warped_corners.min(axis=0)
    x_max, y_max = warped_corners.max(axis=0)
    translation = np.array(
        [[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translated_homography = translation @ homography

    warped_width = max(1, int(math.ceil(x_max - x_min)))
    warped_height = max(1, int(math.ceil(y_max - y_min)))
    warped_image = cv2.warpPerspective(
        image,
        translated_homography,
        (warped_width, warped_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT,
    )
    return warped_image, translated_homography


def image_dir_path(image_dir: Path, sample_name: str) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    return image_dir / f"augmented_{sample_name}.jpg"


def label_dir_path(label_dir: Path, sample_name: str) -> Path:
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / f"augmented_{sample_name}.txt"
