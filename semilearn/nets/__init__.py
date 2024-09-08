# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .resnet import resnet50
from .wrn import wrn_28_2, wrn_28_8, wrn_var_37_2
from .vit import vit_base_patch16_224, vit_small_patch16_224, vit_small_patch2_32, vit_tiny_patch2_32, vit_base_patch16_96
from .bert import bert_base_cased, bert_base_uncased
from .wave2vecv2 import wave2vecv2_base
from .hubert import hubert_base
from .GCC_backbone import GCC_resnet18_cifar,GCC_resnet18_stl,propos_resnet18_cifar1,propos_resnet18_cifar2,propos_resnet18_stl,propos_resnet18_stl_for_size64,propos_resnet34_cifar,propos_resnet34_stl,propos_oriresnet18,propos_resnet50_imagent,SeCu_resnet18_cifar,SeCu_resnet50_imagent
