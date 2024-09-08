"""
Author: Huasong Zhong
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
from PIL import Image
import os
import os.path
import numpy as np
import h5py
from skimage import io, color
import glob
import torchvision.datasets as datasets

from torch.utils.data.dataset import Dataset



import gc
import copy
import json
import random
from torchvision.datasets import ImageFolder
from torchvision import transforms
import math
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation, str_to_interp_mode
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset
from semilearn.datasets.utils import split_ssl_data

mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class getImageNet(Dataset):
    """`ImageNetDogs <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    class_names_file = 'list.txt'
    imagenet_name={'imagenet','imagenet-dogs','imagenet-10','tiny-imagenet-200-new','tiny-imagenet-200'}
    splits = ('train', 'test', 'train+unlabeled')

    def __init__(self, base_folder,root='dataset',split='train',
                 transform=None, target_transform=None, download=False):
        
        if base_folder not in self.imagenet_name:
            raise ValueError('base_folder "{}" not found. Valid splits are: {}'.format(
                base_folder, ', '.join(self.imagenet_name),
            ))
        
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        
        self.base_folder=base_folder
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # train/test/unlabeled set

        # now load the picked numpy arrays
        self.data, self.labels = self.__loadfile()

        class_file = os.path.join(
            self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()

        # consistent with other dataset
        self.targets = self.labels

    def __len__(self):
        return len(self.data)

    def __loadfile(self, data_file=None, labels_file=None):
        datas = []
        labels = []
        data_path = os.path.join(self.root,self.base_folder,self.class_names_file)
        
        with open (data_path, 'r') as fr:
            t=-1
            for line in fr.readlines():
                label=line.strip()
                if self.base_folder=='tiny-imagenet-200':
                    line = os.path.join(self.root,self.base_folder,"train", line.strip(),'images')
                else:
                    line = os.path.join(self.root,self.base_folder,"train", line.strip())
                paths = glob.glob(os.path.join(line, '*.JPEG'))
                t+=1
                #cnt=0
                print(line)
                for path in paths:
                    #import pdb
                    #pdb.set_trace()
                    '''
                    try:
                        img=pil_loader(path)
                    except:
                        print(path)
                    '''
                    
                    datas.append(path)
                    labels.append(t)
                    #cnt+=1
                    #if cnt%50==0:
                    #    print(cnt)
            #import pdb
            #pdb.set_trace()
        return datas, labels

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImagenetDataset(BasicDataset, ImageFolder):
    def __init__(self, data, targets,transform, ulb, alg, strong_transform=None):
        self.alg = alg
        self.is_ulb = ulb
        self.transform = transform
        self.data = data
        self.targets = targets
        self.strong_transform = strong_transform
        self.classes = len(np.unique(targets))
        self.loader = default_loader
        
        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch'], f"alg {self.alg} requires strong augmentation"


    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target



class ImagenetDataset_base(BasicDataset, ImageFolder):
    def __init__(self, root, transform, ulb, alg, strong_transform=None, percentage=-1, include_lb_to_ulb=True, lb_index=None):
        self.alg = alg
        self.is_ulb = ulb
        self.percentage = percentage
        self.transform = transform
        self.root = root
        self.include_lb_to_ulb = include_lb_to_ulb
        self.lb_index = lb_index

        is_valid_file = None
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.data = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]


    def __sample__(self, index):
        path = self.data[index]
        sample = self.loader(path)
        target = self.targets[index]
        return sample, target

    def make_dataset(
            self,
            directory,
            class_to_idx,
            extensions=None,
            is_valid_file=None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return x.lower().endswith(extensions)
        
        lb_idx = {}
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                random.shuffle(fnames)
                if self.percentage != -1:
                    fnames = fnames[:int(len(fnames) * self.percentage)]
                if self.percentage != -1:
                    lb_idx[target_class] = fnames
                for fname in fnames:
                    if not self.include_lb_to_ulb:
                        if fname in self.lb_index[target_class]:
                            continue
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
        gc.collect()
        self.lb_idx = lb_idx
        return instances


def get_GCC_imagenet(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    #data_dir = os.path.join(data_dir, name.lower())
    
    if name != 'imagenet':
        dset=getImageNet(name,data_dir)
    else:
        dset = ImagenetDataset_base(root=os.path.join(data_dir+'/imagenet', "train"), transform=None, alg=alg, ulb=False)
    
    # import random
    # sub_index = random.sample([i for i in range(len(dset.targets))],k=12800)
    # dset.data, dset.targets = np.array(dset.data)[sub_index], np.array(dset.targets)[sub_index]
    data, targets = dset.data, dset.targets
    
    
    
    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    

    lb_dset = ImagenetDataset(lb_data, lb_targets, transform=transform_weak, ulb=False, alg=alg)

    ulb_dset = ImagenetDataset( ulb_data, ulb_targets, transform=transform_weak, alg=alg, ulb=True, strong_transform=transform_strong)
    
    # import pdb
    # pdb.set_trace()
    
    test_data, test_targets = dset.data, dset.targets
    eval_dset = ImagenetDataset(test_data, test_targets, transform=transform_val, alg=alg, ulb=False)
    '''
    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    test_data, test_targets = dset.data, dset.targets
    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)
    '''
    return lb_dset, ulb_dset, eval_dset
    