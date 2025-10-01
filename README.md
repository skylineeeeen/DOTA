
## Requirements 
### Installation
Follow these steps to set up a conda environment and ensure all necessary packages are installed:

```bash
conda create -n dota python=3.7
conda activate dota

# The results are produced with PyTorch 1.12.1 and CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
```

### Dataset
To set up all required datasets, kindly refer to the guidance in [DATASETS.md](docs/DATASETS.md), which incorporates steps for two benchmarks.

## Run dota
### Configs
The configuration for DOTA hyperparameters in `configs/dataset.yaml` can be tailored within the provided file to meet the needs of various datasets. 

### Running

Below are instructions for running DOTA on both Out-of-Distribution (OOD) and Cross-Domain benchmarks using various backbone architectures. Follow the steps suited to your specific needs:"

#### OOD Benchmark
* **ResNet50**: Run DOTA on the OOD Benchmark using the ResNet50 model:
```
bash ./scripts/run_ood_benchmark_rn50.sh 
```
* **ViT/B-16**: Run DOTA on the OOD Benchmark using the ViT/B-16 model.
```
bash ./scripts/run_ood_benchmark_vit.sh 
```

#### Cross-Domain Benchmark
* **ResNet50**: Run DOTA on the Cross-Domain Benchmark using the ResNet50 model:
```
bash ./scripts/run_cd_benchmark_rn50.sh 
```
* **ViT/B-16**: Run DOTA on the Cross-Domain Benchmark using the ViT/B-16 model.
```
bash ./scripts/run_cd_benchmark_vit.sh 
```

