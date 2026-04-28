from __future__ import annotations

import argparse
from pathlib import Path

from src.augment import augment_with_annotations
from src.clean import clean_artifacts, prune_artifacts
from src.config import load_config
from src.eval import evaluate_model
from src.fetch import fetch_dataset
from src.infer import run_inference
from src.setup import prepare_dataset
from src.train import train_model
from src.visualize import launch_visualizer
from src.watch import watch_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO fine-tuning pipeline")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))

    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Download the base dataset archive")
    fetch_parser.add_argument("--url")
    fetch_parser.add_argument("--filename")

    setup_parser = subparsers.add_parser(
        "setup",
        help="Validate source classes and unpack the base dataset into dataset/coco128",
    )
    setup_parser.add_argument("--archive", type=Path)
    setup_parser.add_argument("--force", action="store_true")

    augment_parser = subparsers.add_parser(
        "augment",
        help="Build augmented samples by compositing source classes onto backgrounds",
    )
    augment_parser.add_argument("background_dir", nargs="?", type=Path)
    augment_parser.add_argument("--source-image-dir", type=Path)
    augment_parser.add_argument("--output-dir", type=Path)

    train_parser = subparsers.add_parser(
        "train", help="Train on augmented dataset with curriculum phases"
    )
    train_parser.add_argument("--dataset-yaml", type=Path)
    train_parser.add_argument("--model")
    train_parser.add_argument(
        "--epochs",
        type=int,
        help="Override curriculum main_epochs_per_stage; easy and hard phases still run for 1 epoch each.",
    )
    train_parser.add_argument("--imgsz", type=int)
    train_parser.add_argument("--batch", type=int)
    train_parser.add_argument("--device")
    train_parser.add_argument("--force", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate the latest trained model")
    eval_parser.add_argument("--dataset-yaml", type=Path)
    eval_parser.add_argument("--weights", type=Path)
    eval_parser.add_argument("--force", action="store_true")

    infer_parser = subparsers.add_parser("infer", help="Run inference on a dataset folder")
    infer_parser.add_argument("dataset_dir", nargs="?", type=Path)
    infer_parser.add_argument("--weights", type=Path)
    infer_parser.add_argument("--force", action="store_true")

    watch_parser = subparsers.add_parser("watch", help="Run live inference on a video")
    watch_parser.add_argument("source", nargs="?")
    watch_parser.add_argument("--weights", type=Path)
    watch_parser.add_argument("--conf", type=float)
    watch_parser.add_argument("--imgsz", type=int)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Run fetch, setup, and augment in sequence"
    )
    prepare_parser.add_argument("--url")
    prepare_parser.add_argument("--filename")
    prepare_parser.add_argument("--archive", type=Path)
    prepare_parser.add_argument("--background-dir", type=Path)
    prepare_parser.add_argument("--source-image-dir", type=Path)
    prepare_parser.add_argument("--output-dir", type=Path)
    prepare_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("clean", help="Remove every generated artifact")
    subparsers.add_parser("prune", help="Remove train, eval, infer, and prediction outputs")

    visualize_parser = subparsers.add_parser("visualize", help="Open the dataset visualizer")
    visualize_parser.add_argument("dataset_dir", nargs="?", type=Path)
    visualize_parser.add_argument("--hide-labels", action="store_true")
    visualize_parser.add_argument("--hide-predictions", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "fetch":
        fetch_dataset(config, dataset_url=args.url, archive_name=args.filename)
        return

    if args.command == "setup":
        prepare_dataset(config, archive_path=args.archive, force=args.force)
        return

    if args.command == "augment":
        augment_with_annotations(
            config,
            background_dir=args.background_dir,
            source_image_dir=args.source_image_dir,
            output_dir=args.output_dir,
        )
        return

    if args.command == "train":
        train_model(
            config,
            dataset_yaml_path=args.dataset_yaml,
            model_name=args.model,
            epochs=args.epochs,
            image_size=args.imgsz,
            batch_size=args.batch,
            device=args.device,
            force=args.force,
        )
        return

    if args.command == "eval":
        evaluate_model(
            config,
            dataset_yaml_path=args.dataset_yaml,
            weights_path=args.weights,
            force=args.force,
        )
        return

    if args.command == "infer":
        run_inference(
            config,
            dataset_subdir=args.dataset_dir,
            weights_path=args.weights,
            force=args.force,
        )
        return

    if args.command == "prepare":
        fetch_dataset(config, dataset_url=args.url, archive_name=args.filename)
        prepare_dataset(config, archive_path=args.archive, force=args.force)
        augment_with_annotations(
            config,
            background_dir=args.background_dir,
            source_image_dir=args.source_image_dir,
            output_dir=args.output_dir,
        )
        return

    if args.command == "watch":
        watch_video(
            config,
            source=args.source,
            weights_path=args.weights,
            confidence=args.conf,
            image_size=args.imgsz,
        )
        return

    if args.command == "clean":
        clean_artifacts(config)
        return

    if args.command == "prune":
        prune_artifacts(config)
        return

    if args.command == "visualize":
        launch_visualizer(
            config,
            dataset_subdir=args.dataset_dir,
            show_labels=not args.hide_labels,
            show_predictions=not args.hide_predictions,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
