# AGENTS.md — yolo-fine-tuning-pipeline

## Quick Start

```bash
just install          # uv sync — creates .venv
just prepare          # fetch + setup + augment
just augment [bg_dir] # composites images from images/<class>/ onto backgrounds
just train            # builds dataset/train, trains model, logs to Trackio
just eval             # evaluates latest trained weights
just infer            # runs inference on dataset/augmented (default)
just visualize        # opens dataset visualizer
just prune            # removes train/eval/infer artifacts and predictions
just clean            # removes entire generated dataset tree
```

## Toolchain

- **Package manager**: `uv` (lockfile `uv.lock` is checked in)
- **Task runner**: `just` (see `Justfile`)
- **Python**: >=3.12,<3.13
- **Entry point**: `cli.py` — all commands are `uv run python cli.py <command> [args]`
- **Config**: `config.toml` is the single source of truth. CLI flags override config values. Pass `--config <path>` to use a different config.

## PyTorch / CUDA

- `pyproject.toml` configures an explicit `pytorch-cu128` index for `torch` and `torchvision`. Reinstalling the env picks up CUDA 12.8 wheels automatically.
- Device resolution in `src/train.py`: `"auto"` → CUDA if available, else CPU. `"0"` or `"cuda"` raises if CUDA unavailable. Workers forced to 0 on Windows.

## Dataset Layout

All dataset subdirectories follow the same structure:

```
dataset/
  raw/            # downloaded archive
  coco128/        # unpacked base dataset (background images)
  augmented/      # synthetic composites (output of `just augment`)
  train/          # assembled train/val split + weights (output of `just train`)
  eval/           # evaluation results
  infer/          # inference predictions
```

Each dataset folder contains `images/<split>/`, `labels/<split>/`, `predictions/<split>/`, `classes.json`, `dataset.yaml`, `manifest.json`. Splits are `train2017` and `val2017` (defined in `src/common.py`).

## Source Images

The `images/` directory (configured via `augment_source_dir` in `config.toml`) holds class-specific images organized by subfolder:

```
images/
  panda/
    img1.jpg
    img2.jpg
  mouse/
    img3.jpg
```

Subfolder names become class names. No manual annotation is needed — classes are inferred from the directory structure.

## Pipeline Order Matters

Commands depend on prior steps producing artifacts:
1. `fetch` → `dataset/raw/coco128.zip`
2. `setup` → `dataset/coco128/`
3. `augment [bg_dir]` → `dataset/augmented/` (required before `train`)
4. `train` → `dataset/train/` + weights at `dataset/train/best.pt` and `dataset/train/latest.pt`
5. `eval` / `infer` / `watch` → consume trained weights

`just prepare` combines fetch + setup + augment. `augment` and `prepare`
default to `[augment].background_dir` from `config.toml`.

## Augmentation

The augment step composites source images onto background images:
- Source images come from `images/<class_name>/` subfolders
- Class labels are inferred from subfolder names
- Each background image gets N objects placed at random positions (N from `min_objects`..`max_objects` in config)
- `min_class_appearances` controls baseline coverage. A value of `10` makes each class the required object in at least 10 separate composite images. When this requires more samples than the available background count, backgrounds are reused with new random placement and scale.
- Extra objects are selected from the least-used classes so class selection stays balanced across the output dataset.
- Object scale is randomized between `scale_min` and `scale_max` (fraction of background's shorter dimension)
- YOLO labels are generated from the clipped pasted pixel rectangle and normalized to YOLO xywh format
- Output is split into train/val using `setup.train_split` and `setup.random_seed`, with validation samples chosen greedily for class coverage before filling remaining slots.

## src/ Module Map

| File | Purpose |
|---|---|
| `cli.py` | Argument parser, dispatches to src/ modules |
| `src/config.py` | Loads `config.toml` into frozen dataclasses |
| `src/common.py` | Shared utilities: path resolution, YOLO label I/O, class maps, dataset YAML generation |
| `src/fetch.py` | Downloads dataset archive |
| `src/setup.py` | Unpacks archive, creates initial split |
| `src/augment.py` | Composites source images onto backgrounds with random placement and scaling |
| `src/train.py` | Builds training dataset from augmented samples, runs ultralytics YOLO training with Trackio callbacks |
| `src/eval.py` | Model evaluation |
| `src/infer.py` | Batch inference |
| `src/watch.py` | Live video inference |
| `src/tracking.py` | Trackio experiment logging |
| `src/visualize.py` | Dataset visualizer |
| `src/clean.py` | Artifact cleanup |

## No Tests, Lint, or CI

This repo has no test suite, linter, formatter, type checker, or CI workflows. Do not assume one exists.

## Key Gotchas

- `dataset/` is gitignored. All pipeline output lives there and is ephemeral.
- Run names are auto-incremented via `dataset/run_versions.json`. Training refuses to overwrite an existing run unless `--force` is passed.
- The training dataset is rebuilt from scratch each time `train` runs (augmented samples are re-split).
- Curriculum epochs run inside one continuous Ultralytics trainer. The trainer updates augmentation transforms at epoch boundaries, preserving live optimizer/scaler/EMA/scheduler state and the resolved auto batch size. Augmentation range values are interpolated per epoch across `stages * epochs_per_stage` total epochs, using `epoch_index / total_epochs`.
- `config.toml` paths are resolved relative to the config file's parent directory (project root).
