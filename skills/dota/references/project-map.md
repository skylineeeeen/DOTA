# DOTA Project Map

Use this map when navigating, modifying, or explaining the DOTA repository.

## Core Files

- `dota.py`: main DOTA evaluation/adaptation entrypoint.
- `utils.py`: shared helpers for config loading, CLIP classifier construction, accuracy, entropy, and data loader wiring.
- `requirements.txt`: Python package dependencies installed after the conda/PyTorch setup from `README.md`.
- `README.md`: public setup and benchmark instructions.
- `docs/DATASETS.md`: public dataset preparation guide.

## Model and Data Code

- `clip/`: bundled CLIP implementation and tokenizer files.
- `datasets/`: dataset wrappers and dataset registry.
- `datasets/__init__.py`: maps command-line dataset aliases to dataset classes.
- `configs/vit/`: per-dataset DOTA hyperparameter YAML files.

## Benchmark Scripts

- `scripts/run_ood_benchmark_vit.sh`: ViT-B/16 OOD benchmark command.
- `scripts/run_cd_benchmark_vit.sh`: ViT-B/16 cross-domain benchmark command.

Inspect `scripts/` before mentioning other benchmark scripts, because the README may describe scripts that are not present in a partial public release.

## Main Entrypoint Arguments

`dota.py` accepts:

- `--config`: config directory, default `configs`; current scripts use `configs/vit`.
- `--datasets`: slash-separated dataset aliases, for example `I/A/V/R/S` or `caltech101/dtd/eurosat`.
- `--data-root`: dataset root directory, default `./dataset/`.
- `--backbone`: CLIP backbone, currently constrained to `ViT-B/16`.
- `--log-path`: log directory, default `./log`.

## Logs

Each dataset run creates a log filename from the backbone, dataset name, and timestamp. Standard scripts write logs under `./log`.
