# CASS-ADM
![CASS_ADM](./figures/CASS-ADMS.jpg)
**CASS-ADM**: A Controlled Anomaly Synthesis Strategy with Asymptotic Diffusion Modulation for Image Anomaly Detection and Localization

# Introduction
This repository provides the PyTorch implementation of two CASS-ADM variants for image anomaly detection and localization. The CASS-ADM series employs a controllable feature-level anomaly synthesis method, incorporating three core components: a **Noise Distributor**, a **Gradient-Guided Direction Awareness module**, and an **Adaptive Direction Regulation module**, to achieve decoupling control of feature-level noise intensity and direction.

**CASS-ADMS**: a lightweight version that uses only the feature-level anomaly synthesis strategy and does not require any external data sources.

**CASS-ADMM**: a dual-level version that combines image-level and feature-level anomaly synthesis and leverages the external DTD texture dataset, and performs image-level anomaly synthesis without using foreground masks.

Both variants are built on the same network architecture.

Performance on Public Datasets
| Metric   | MVTec AD | VisA  | MPDD  | MVTec LOCO  |
|----------|----------|-------|-------|-------------|
| I-AUROC  | 99.8%    | 97.8% | 98.7% | 82.8%       |
| P-AUROC  | 99.2%    | 98.8% | 99.3% | 87.1%       | 
|  AUPRO   | 96.7%    | 94.0% | 97.6% | 70.3%       | 

Resource Consumption on an NVIDIA RTX 4080 Super GPU
| Dataset  | Batch Size | Epochs | Train Time (h) | GPU Memory (GB) | Throughput (FPS) |
|----------|------------|--------|----------------|-----------------|------------------|
| MVTec AD | 8          | 200    | 25             | 3.3             | 82               |

# Data Preparation
DTD is an auxiliary texture dataset used only for training CASS-ADMM, while the other datasets are used for anomaly detection evaluation. In addition, a few-shot steel cross-section anomaly detection dataset is introduced in this work.
- [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [VisA](https://github.com/amazon-science/spot-diff/)
- [MPDD](https://github.com/stepanje/MPDD/)
- [MVTec LOCO](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)
- [Steel AD](https://drive.google.com/file/d/15kf6COvGeDcWT-1ihMsVEe86BG5eIsPf/view?usp=drive_link)
  
Please keep the dataset folders in their original directory structures.

# Environment

```python
conda create -n cassadm python=3.11.15
conda activate cassadm
pip install -r requirements.txt
```
Our experiments were conducted on an RTX 4080 Super GPU. Please use the same configuration whenever possible.

# Run on MVTec-AD
Edit `./shell/run_mvtec.sh` to configure arguments `--datapath`, `--augpath`, `--classes`, and `--test`.

Set `--augpath` to `None` for CASS-ADMS and set `--augpath` to the path of an external texture dataset for CASS-ADMM.

Set `--test` to an empty string (`''`) for training, and set it to the checkpoint filename (e.g., `best_roc.pth`) for testing.

Qualitative results of **CASS-ADMS (Row 3)** and **CASS-ADMM (Row 4)** on the MVTec AD dataset.

![mvtec_ad](./figures/mvtec_ad.jpg)

# Run on Steel-AD 

Qualitative results of different models on the Steel-AD dataset in the **1-shot** setting. From top to bottom, the rows show the input image, anomaly mask, and experimental results.
 
![steel_ad](./figures/1-shot_results.jpg)

# Acknowledgements
We gratefully acknowledge the inspiration provided by [SimpleNet](https://github.com/DonaldRR/SimpleNet/) and [GLASS](https://github.com/cqylunlun/GLASS#data-preparation). 


# License
All code within the repo is under [MIT license](https://mit-license.org/)

