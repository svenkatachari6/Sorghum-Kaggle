import random
import os
import torch
import torch.utils.data
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import pandas as pd
import numpy as np
import csv


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def standardization(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    for t in tensor:
        t.sub_(t.mean()).div_(t.std())
    
    # assert not torch.sum(np.isnan(tensor)), 'Input value = NaN'
        
    return tensor


def l2normalize(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    tensor = tensor.mul(255)
    norm_tensor = tensor/torch.norm(tensor)
    
    # assert not torch.sum(np.isnan(norm_tensor)), 'Input value = NaN'
        
    return norm_tensor

def make_dataset (traindir, if_train = True):
    file_path = os.path.join(traindir, 'train_cultivar_mapping.csv')
    img = []
    cnt = 0
    target_dict = {}
    target_cnt = 0
    with open(file_path, 'r') as f:
        f.readline()
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            target_name = line[1]
            if target_name in target_dict:
                pass
            else:
                target_dict[target_name] = target_cnt
                target_cnt += 1
            path = os.path.join(traindir, line[0])
            item = (path, target_dict[target_name])
            if target_dict[target_name]<=2:
                img.append(item)
                cnt += 1
    return img


class SorghumDataset (torch.utils.data.Dataset):
    def __init__(self, traindir, transform=None, target_transform =None,
                loader = pil_loader, if_train = True):
        self.traindir = traindir
        self.imgs = make_dataset (traindir, if_train)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)