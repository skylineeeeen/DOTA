# Running DOTA

Use this guide when running benchmarks, changing commands, or troubleshooting execution.

## Environment

The public README recommends:

```bash
conda create -n dota python=3.7
conda activate dota
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Do not run installation commands unless the user asks. When diagnosing environment issues, check the active Python, CUDA, PyTorch, and package versions first.

## Standard Runs

OOD benchmark with ViT-B/16:

```bash
bash ./scripts/run_ood_benchmark_vit.sh
```

Cross-domain benchmark with ViT-B/16:

```bash
bash ./scripts/run_cd_benchmark_vit.sh
```

The current public scripts set `CUDA_VISIBLE_DEVICES=0`, use `--config configs/vit`, write logs to `./log`, and use `--backbone ViT-B/16`.

## Custom Runs

Use `dota.py` directly for custom dataset groups:

```bash
CUDA_VISIBLE_DEVICES=0 python dota.py \
  --config configs/vit \
  --data-root ./dataset/ \
  --log-path ./log \
  --datasets caltech101/dtd/eurosat \
  --backbone ViT-B/16
```

`--datasets` must be slash-separated. Each name must resolve through dataset loading and config loading. Check `datasets/__init__.py` and `configs/vit/` together when changing this list.

## Troubleshooting Checklist

- If a dataset name fails, compare the command alias with `datasets/__init__.py` and `configs/vit/`.
- If config loading fails, verify that the dataset name has a matching YAML file under the selected config directory.
- If files are missing, read `docs/DATASETS.md` and verify the expected `$DATA` layout.
- If CUDA or CLIP loading fails, confirm that the active environment matches the README's PyTorch/CUDA guidance.
- If logs are missing, ensure `--log-path` exists or create it before running.
