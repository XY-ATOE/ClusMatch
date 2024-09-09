<div id="top"></div>

## Introduction




## Preparing Environment

use pip to install required packages:

```
pip install -r requirements.txt
```

## Preparing Pre-trained Model

Our method requires pre-trained models and there are some training logs and pretrained models.
[Google Drive](https://drive.google.com/drive/folders/1yGhOTJFkF0pSr2m_vwOnhiWtVmzsXPTE?usp=drive_link)


## Training

```sh
# ProPos+CIFAR-10+ResNet-18
python train.py --c config/classic_cv/clusmatch/clusmatch_cifar10_600_0_propos.yaml

# ProPos+CIFAR-100+ResNet-18
python train.py --c config/classic_cv/clusmatch/clusmatch_cifar100_6000_0_propos.yaml
```
Please check if the "load_path" is correct before training.
