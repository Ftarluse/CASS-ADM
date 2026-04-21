# CASS-AMD
![CASS_ADM](https://github.com/user-attachments/assets/b900f728-101b-4860-a179-53dced760c52)
CASS-ADM: A Controlled Anomaly Synthesis Strategy with Asymptotic Diffusion Modulation for Image Anomaly Detection and Localization

# Introduction
This repository contains source code for two CASS-ADM variants implemented with PyTorch. The CASS-ADM series adopts a controllable feature-level anomaly synthesis method, incorporating three core components: a progressive noise allocator, a gradient-guided direction perceiver, and an adaptive directional anomaly modulator, to achieve decoupling control of feature-level noise intensity and direction.

**CASS-ADMS**: a lightweight version that uses only the feature-level anomaly synthesis strategy and does not require any external data sources.

**CASS-ADMM**: a dual-level version that combines image-level and feature-level anomaly synthesis and leverages the external **DTD** texture dataset.

Both variants are built on the same network architecture.


