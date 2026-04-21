# CASS-AMD
![CASS_ADM](./figures/CASS-ADMS.jpg)
CASS-ADM: A Controlled Anomaly Synthesis Strategy with Asymptotic Diffusion Modulation for Image Anomaly Detection and Localization

# Introduction
This repository contains source code for two CASS-ADM variants implemented with PyTorch. The CASS-ADM series employs a controllable feature-level anomaly synthesis method, incorporating three core components: a Noise Distributor, a Gradient-Guided Direction Awareness module, and an Adaptive Direction Regulation module, to achieve decoupling control of feature-level noise intensity and direction.

**CASS-ADMS**: a lightweight version that uses only the feature-level anomaly synthesis strategy and does not require any external data sources.

**CASS-ADMM**: a dual-level version that combines image-level and feature-level anomaly synthesis and leverages the external **DTD** texture dataset.

Both variants are built on the same network architecture.

# Environments

```python
conda create -n CASS_ADM python=3.11.15
conda activate CASS_ADM
pip install -r requirements.txt
```

# Run
Edit ./shell/run-.sh to configure arguments
