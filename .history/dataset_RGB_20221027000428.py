import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class DataLoaderTrain(Dataset):
    def __init__(self,xs, sigma,img_options=None) -> None:
        super(DataLoaderTrain, self).__init__()
        self.img_options = img_options
        self.xs = xs
        self.ps = self.img_options['patch_size']
        self.sigma = sigma
    
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        index_ = index % len(self.xs)
        ps = self.ps
        tar_img =  torch.from_numpy(self.xs[index_])
        noise   = torch.randn(tar_img.size()).mul_(random.randint(10,50)/255.0)
        inp_img = tar_img + noise

        _,w, h = tar_img.size()
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr  = random.randint(0, hh - ps)
        cc  = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]
        ## Call augmentation function 
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))
        return tar_img, inp_img


class DataLoaderVal(Dataset):
    def __init__(self, xs, sigma, img_options=None):
        super(DataLoaderVal, self).__init__()
        self.xs = xs
        self.sigma = sigma
        self.img_options = img_options
        self.ps = self.img_options['patch_size']

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        index_ = index %  len(self.xs)
        ps = self.ps
        tar_img =  torch.from_numpy(self.xs[index_])
        noise   = torch.randn(tar_img.size()).mul_(random.randint(10,50)/255.0)
        inp_img = tar_img + noise

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps, ps))
            tar_img = TF.center_crop(tar_img, (ps, ps))

        return tar_img, inp_img, ""

class DataLoaderTest(Dataset):
    def __init__(self, xs, sigma, img_options=None):
        super(DataLoaderTest, self).__init__()
        self.xs = xs
        self.sigma = random.randint(10,50)
        self.img_options = img_options

    def __len__(self):
        return  len(self.xs)

    def __getitem__(self, index):
        index_ = index %  len(self.xs)
        tar_img = torch.from_numpy(self.xs[index_])
        noise   = torch.randn(tar_img.size()).mul_(self.sigma/255.0)
        inp_img = tar_img + noise

        return tar_img, inp_img, ""
