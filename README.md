# UaD-ClE

This repository contains the PyTorch implementation for TNNLS paper
"Uncertainty-Aware Distillation for Semi-Supervised Few-Shot Class-Incremental Learning" 

The code is based on [CVPR19_Incremental_Learning](https://github.com/hshustc/CVPR19_Incremental_Learning).

## Running environment

Pytorch 1.10 with cuda 10.0 and cudnn 7.4 in Ubuntu 18.04 system <br>

## Dataset

We follow  [FSCIL](https://github.com/xyutao/fscil) setting to use the same data index_list for training.

For CUB200, you can download from [this link](https://drive.google.com/file/d/13_PsB0dP_SuNrqwhNQCnfoJD-z-6_jGw/view?usp=sharing). Please put the downloaded file under data/cub folder and unzip it.

## Checkpoint

We provide a trained model of the first session. Please download from [this link](https://drive.google.com/file/d/1IOoD8EoPRoYA285bYgOZ7FoWEiJFuK_E/view?usp=sharing) and put it in ./checkpoint.

## Training script for CUB200

```bash
# train, the checkpoints will be save in ./checkpoint
CUDA_VISIBLE_DEVICES=0 python train_cub.py --resume --uncertainty_distillation --frozen_backbone_part --flip_on_means --adapt_lamda

##Citation
@article{cui2023uncertainty,
  title={Uncertainty-Aware Distillation for Semi-Supervised Few-Shot Class-Incremental Learning},
  author={Cui, Yawen and Deng, Wanxia and Chen, Haoyu and Liu, Li},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023}
  publisher={IEEE}
}
