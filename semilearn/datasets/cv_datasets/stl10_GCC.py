# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math 
from torchvision import transforms

from .datasetbase import BasicDataset
from semilearn.datasets.utils import sample_labeled_unlabeled_data,sample_labeled_unlabeled_data_from_selfsup
from semilearn.datasets.augmentation import RandAugment


mean, std = {}, {}
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
img_size = 96

def get_transform(mean, std, crop_size, train=True, crop_ratio=0.95):
    img_size = int(img_size / crop_ratio)

    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.Resize(img_size),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.Resize(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


def get_stl10_GCC(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=False):
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    img_size = int(math.floor(crop_size / crop_ratio))

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    dset = getattr(torchvision.datasets, name.upper())
    dset_lb = dset(data_dir, split='train', download=True)
    dset_test = dset(data_dir, split='test', download=True)
    lb_data, lb_targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
    lb_data_test, lb_targets_test = dset_test.data.transpose([0, 2, 3, 1]), dset_test.labels.astype(np.int64)


    # Note this data can have imbalanced labeled set, and with unknown unlabeled set
    ulb_data = np.concatenate([lb_data, lb_data_test], axis=0)
    ulb_targets = np.concatenate([lb_targets, lb_targets_test], axis=0)
    lb_idx, _ = sample_labeled_unlabeled_data_from_selfsup(args, lb_data, lb_targets, num_classes,
                                              lb_num_labels=num_labels,
                                              ulb_num_labels=args.ulb_num_labels,
                                              lb_imbalance_ratio=args.lb_imb_ratio,
                                              ulb_imbalance_ratio=args.ulb_imb_ratio,
                                              load_exist=True)
    
    lb_data, lb_targets = ulb_data[lb_idx], ulb_targets[lb_idx]
    

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    data, targets = ulb_data,ulb_targets
    eval_dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset
