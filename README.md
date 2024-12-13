# Continual Source-Free Active Domain Adaptation for Nasopharyngeal Carcinoma Tumor Segmentation across Multiple Hospitals
This repository contains the official implementation of our paper. 

## 1. Training source models in a labeled source domain dataset.

## 2. Training target models in a unlabeled target domain dataset by using the $\mathcal{L}_{\text{CCL}}$.
The implementation details of the $\mathcal{L}_{\text{CCL}}$ can be found in the `cc_loss.py` file.

## 3. Select Active Samples by using the proposed method.
The implementation details of the proposed method can be found in the `active_sample_selection.py` file.
Code will be released soon.

## 4. Fine-tuning the target model with the selected active samples.
The implementation details of the $\mathcal{L}_{\text{CCNTD}}$ can be found in the `ccntd_loss.py` file.

# Dataset 📊
We use the multi-center NPC T1-weighted MRI dataset released by [here](https://ieeexplore.ieee.org/abstract/document/10553522). This dataset is the first publicly available multi-center GTV segmentation dataset for research purposes, focusing on patients diagnosed with NPC. The dataset contains annotated GTV data obtained from three different medical institutions (CenterA, CenterB, and CenterC). 
The dataset includes 50, 50, and 60 patients from CenterA, CenterB, and CenterC, respectively, each associated with a single T1-weighted MRI image. 
In our experiments, for each center, the dataset is divided into training, validation, and test sets at a ratio of 7:1:2.

# Comparison with Other Methods 📈

We acknowledge the developers of the comparative methods in [ADA4MIA](https://github.com/whq-xxh/ADA4MIA) , [STDR](https://github.com/whq-xxh/Active-GTV-Seg) and  [CCL/NTD](https://github.com/WenkeHuang/FCCL).
