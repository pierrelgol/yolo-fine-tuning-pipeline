from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.common import resolve_path
from src.config import AppConfig


def watch_video(
    config: AppConfig,
    source: str | None = None,
    weights_path: Path | None = None,
    confidence: float | None = None,
    image_size: int | None = None,
) -> None:
    import cv2

    selected_source = resolve_watch_source(config, source)
    selected_weights_path = (
        resolve_path(weights_path, base_dir=config.paths.project_root)
        if weights_path is not None
        else config.paths.train_best_weights_path
    )
    selected_confidence = config.watch.confidence if confidence is None else confidence
    selected_image_size = config.watch.image_size if image_size is None else image_size

    if not selected_weights_path.exists():
        raise FileNotFoundError(f"Watch weights not found: {selected_weights_path}")

    capture_source = parse_capture_source(selected_source)
    video_capture = cv2.VideoCapture(capture_source)
    if not video_capture.isOpened():
        raise FileNotFoundError(f"Unable to open video source: {selected_source}")

    model = YOLO(str(selected_weights_path))
    window_name = config.watch.window_name
    paused = False

    print(f"Watching source: {selected_source}")
    print(f"Weights: {selected_weights_path}")
    print("Controls: press q to quit, space to pause/resume.")

    try:
        while True:
            if not paused:
                has_frame, frame = video_capture.read()
                if not has_frame:
                    break

                result = model.predict(
                    source=frame,
                    conf=selected_confidence,
                    imgsz=selected_image_size,
                    device=resolve_watch_device(config.train.device),
                    verbose=False,
                )[0]
                rendered_frame = result.plot()
                draw_controls_overlay(rendered_frame, selected_source)
                cv2.imshow(window_name, rendered_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        video_capture.release()
        cv2.destroyAllWindows()


def resolve_watch_source(config: AppConfig, source: str | None) -> str:
    selected_source = source or config.watch.source
    if selected_source.strip():
        return selected_source.strip()
    raise ValueError("No video source provided. Pass a path to `watch` or set [watch].source in config.toml.")


def parse_capture_source(source: str) -> str | int:
    stripped_source = source.strip()
    if stripped_source.isdigit():
        return int(stripped_source)
    return stripped_source


def resolve_watch_device(configured_device: str) -> str:
    normalized_device = configured_device.strip().lower()
    if normalized_device in {"", "auto", "cuda", "cuda:0"}:
        return "0"
    if normalized_device == "cpu":
        return "cpu"
    if normalized_device.startswith("cuda:"):
        return normalized_device.split(":", maxsplit=1)[1]
    return normalized_device


def draw_controls_overlay(frame, source: str) -> None:
    import cv2

    source_name = Path(source).name or source
    cv2.putText(frame, f"source: {source_name}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "q: quit  space: pause", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
