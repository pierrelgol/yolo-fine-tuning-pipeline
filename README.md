# yolo-fine-tuning-pipeline

Minimal, beginner-friendly YOLO fine-tuning pipeline.

Everything is driven by `config.toml` and `cli.py`. Each step stays in its own script under `src/`, and the dataset layout stays consistent across the whole pipeline:

```text
dataset/
  raw/
  coco128/
  augmented/
  train/
  eval/
  infer/
```

Each dataset folder uses the same basic shape when relevant:

```text
<dataset>/
  images/train2017/
  images/val2017/
  labels/train2017/
  labels/val2017/
  predictions/train2017/
  predictions/val2017/
  classes.json
  dataset.yaml
  manifest.json
```

Source class images live outside `dataset/` and are ignored by git:

```text
images/
  P01-V/
    P01-V.png
  T20-V2/
    T20-V2.png
```

Each direct subfolder of `images/` is treated as one class.

## Install

```text
just install
```

This project is intentionally CUDA-first on Windows through the PyTorch CUDA index configured in `pyproject.toml`.

## Typical Flow

```text
just prepare
just train
just eval
just infer
just watch <video_path>
just show
just visualize
```

## Commands

- `just fetch`: download the base dataset archive into `dataset/raw`
- `just setup`: validate `images/<class_name>/`, unpack the archive into `dataset/coco128`, and write source class metadata
- `just augment [background_dir]`: build a baseline synthetic detection dataset in `dataset/augmented`
- `just prepare`: run fetch, setup, and augment in sequence
- `just train`: build `dataset/train`, run curriculum fine-tuning, save stable weights, log to Trackio
- `just eval`: evaluate the latest trained weights
- `just infer`: run inference on the augmented validation split by default
- `just watch <video_path>`: open a live video window with YOLO predictions
- `just show`: open the Trackio UI
- `just visualize`: inspect labels and predictions, defaulting to the latest infer target
- `just prune`: remove train/eval/infer artifacts and predictions
- `just clean`: remove the full generated dataset tree

## Configuration

`config.toml` is the single source of truth for:

- paths
- fetch source
- baseline compositing settings
- train defaults
- train optimizer hyperparameters
- train curriculum ranges
- eval defaults
- infer defaults
- Trackio logging frequency and limits

Override any command from the CLI when needed:

```text
uv run python cli.py train --epochs 2 --batch 16
```

`--epochs` overrides `train.curriculum.epochs_per_stage`. Every augmentation range is linearly interpolated once per epoch across `train.curriculum.stages * train.curriculum.epochs_per_stage` total curriculum epochs. For example, a range `[0, 12]` over 12 total epochs produces values equivalent to `0, 1, ..., 11`.

`prepare` and `augment` default to `[augment].background_dir`, which is `dataset/coco128/images/train2017` in the checked-in config.
`augment.min_class_appearances` controls the baseline dataset size. A value of `10` means every class is used as the required object in at least 10 separate composite images. If that requires more composites than the available background count, backgrounds are reused with new random placements and scales.

Curriculum epochs run inside one continuous Ultralytics trainer. The trainer updates augmentation transforms at epoch boundaries, so optimizer, scaler, EMA, scheduler progress, and the auto-selected batch size remain live until the full curriculum finishes.
