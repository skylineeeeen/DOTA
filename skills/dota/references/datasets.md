# DOTA Datasets

Use this guide when preparing datasets, editing benchmark dataset lists, or debugging dataset/config mismatches.

## Data Root

The code defaults to `--data-root ./dataset/`. The public docs describe a shared `$DATA` directory containing dataset folders such as:

- `imagenet/`
- `caltech-101/`
- `oxford_pets/`
- `stanford_cars/`

If the user's datasets live elsewhere, pass that location with `--data-root` or create symlinks under the expected root.

## Dataset Aliases

The registry in `datasets/__init__.py` maps these command aliases:

- `caltech101`
- `dtd`
- `eurosat`
- `fgvc`
- `food101`
- `oxford_flowers`
- `oxford_pets`
- `stanford_cars`
- `sun397`
- `ucf101`
- `imagenet-a`
- `imagenet-v`
- `imagenet-r`
- `imagenet-s`

The OOD script uses compact aliases handled in `utils.py`:

- `I`: loads ImageNet and `configs/vit/imagenet.yaml`.
- `A`: loads `imagenet-a` and `configs/vit/imagenet_a.yaml`.
- `V`: loads `imagenet-v` and `configs/vit/imagenet_v.yaml`.
- `R`: loads `imagenet-r` and `configs/vit/imagenet_r.yaml`.
- `S`: loads `imagenet-s` and `configs/vit/imagenet_s.yaml`.

## Config Files

ViT configs live in `configs/vit/`. Current files include:

- `caltech101.yaml`
- `dtd.yaml`
- `eurosat.yaml`
- `fgvc.yaml`
- `food101.yaml`
- `imagenet.yaml`
- `imagenet_a.yaml`
- `imagenet_r.yaml`
- `imagenet_s.yaml`
- `imagenet_v.yaml`
- `oxford_flowers.yaml`
- `oxford_pets.yaml`
- `stanford_cars.yaml`
- `sun397.yaml`
- `ucf101.yaml`

When adding a dataset, update the dataset wrapper, the registry alias, and the config file together.

## Public Dataset Documentation

Use `docs/DATASETS.md` for the authoritative directory layouts and download sources. It covers ImageNet, Caltech101, OxfordPets, StanfordCars, Flowers102, Food101, FGVCAircraft, SUN397, DTD, EuroSAT, UCF101, ImageNetV2, ImageNet-Sketch, ImageNet-A, and ImageNet-R.
