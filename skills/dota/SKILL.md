---
name: dota
description: Work with the DOTA research codebase for dataset setup, benchmark execution, configuration changes, reproduction guidance, and troubleshooting. Use when an AI coding agent needs to run or modify DOTA experiments, explain the repository layout, prepare OOD or cross-domain benchmarks, edit dataset/config mappings, inspect logs, or debug DOTA training/evaluation failures.
---

# DOTA

## Overview

Use this skill to operate the public DOTA repository as a research codebase. Keep changes compatible with the existing entrypoint, scripts, config layout, and dataset conventions.

## Repository

Assume the repository root is the current working directory when invoked from this repo. If the user provides an absolute clone path, work from that path instead.

Start by reading only the references needed for the task:

- For repository structure and key files, read `references/project-map.md`.
- For running benchmarks or changing command lines, read `references/running.md`.
- For dataset preparation, aliases, or config mappings, read `references/datasets.md`.

## Workflow

1. Inspect the task and identify whether it concerns environment setup, dataset layout, benchmark execution, code changes, or troubleshooting.
2. Read the relevant reference file before editing or running commands.
3. Prefer the existing shell scripts for standard benchmark runs.
4. Use `dota.py` directly only when the user needs a custom dataset list, config path, data root, log path, or device selection.
5. Keep benchmark command examples faithful to the current code: `--datasets` uses slash-separated names, `--config` points to a config directory, `--data-root` points to the dataset root, and `--log-path` points to a directory for log files.
6. Before changing public docs or scripts, verify the corresponding dataset alias exists in `datasets/__init__.py` and that a matching YAML file exists under `configs/vit/`.

## Common Commands

Run the ViT OOD benchmark:

```bash
bash ./scripts/run_ood_benchmark_vit.sh
```

Run the ViT cross-domain benchmark:

```bash
bash ./scripts/run_cd_benchmark_vit.sh
```

Run a custom command:

```bash
CUDA_VISIBLE_DEVICES=0 python dota.py \
  --config configs/vit \
  --data-root ./dataset/ \
  --log-path ./log \
  --datasets caltech101/dtd/eurosat \
  --backbone ViT-B/16
```

## Guardrails

- Do not download datasets or install dependencies unless the user explicitly asks.
- Do not assume ResNet scripts exist; inspect `scripts/` before citing a command.
- Do not rename dataset aliases casually, because aliases are used by config lookup, dataset loading, and benchmark scripts.
- Preserve the repository's current Python and shell style unless a task specifically asks for modernization.
