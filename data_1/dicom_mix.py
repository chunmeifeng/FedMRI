import pydicom
import os
import torch
import numpy as np
import random

from scipy.io import loadmat
from .transforms import center_crop,normalize_instance, normalize
from torch.utils.data import Dataset
from data.transforms import to_tensor
from matplotlib import pyplot as plt
from .math import *
def fft2(img):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    return slices
def vis_img(img, fname, ftype ,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(img, cmap='gray')
    figname = fname + '_' + ftype + '.png'
    figpath = os.path.join(output_dir, figname)
    plt.savefig(figpath)
class FastMRIDicom_Split(Dataset):
    def __init__(self, list_file, mask, crop_size=(192, 192), client_idx = 0, client_num = 2,mode='train',sample_rate=1):
        paths = []
        with open(list_file) as f:
            for line in f:
                path = line.strip()
                name = line.split('/')[-1]
                for slice in range(0, 32):
                    paths.append((path, slice, name))

        self.paths = paths
        self.crop_size = crop_size
        self.mask = loadmat(mask[client_idx])['mask']

        if sample_rate < 1:
            if mode == 'train':
                random.shuffle(paths)
            num_examples = round(len(paths) * sample_rate)
            paths = paths[0:num_examples]

        split_paths = {}
        for i in range(client_num):
            split_paths[i] = []

        for i in range(len(paths)):
            split_paths[i % client_num].append(paths[i])

        self.examples = split_paths[client_idx]


    def __getitem__(self, item):
        path, slice, fname = self.examples[item]

        img_path = os.path.join(path, str(slice)+'.mat')
        img_path = img_path.replace('fastMRI_brain_DICOM', 'fastMRI_brain_DICOM_mat')
        img = loadmat(img_path)['img']


        kspace = fft2(img)
        kspace = center_crop(kspace, self.crop_size)
        maskedkspace = kspace * self.mask

        subsample = abs(np.fft.ifft2(maskedkspace))
        target = abs(np.fft.ifft2(kspace))

        subsample, mean, std = normalize_instance(subsample, eps=1e-11)
        target = normalize(target, mean, std, eps=1e-11)

        subsample = torch.from_numpy(subsample).float()
        target = torch.from_numpy(target).float()

        return subsample, target, mean, std, fname, slice

    def __len__(self):
        return len(self.examples)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

class FastMRIDicom(Dataset):
    def __init__(self, list_file, mask, crop_size=(192, 192), mode='train', sample_rate=1):
        paths = []
        with open(list_file) as f:
            for line in f:
                path = line.strip()
                name = line.strip().split('/')[-1]
                for slice in range(0, 32):
                    paths.append((path, slice, name))

        self.crop_size = crop_size
        self.mask = loadmat(mask)['mask']

        if sample_rate < 1:
            if mode == 'train':
                random.shuffle(paths)
            num_examples = round(len(paths) * sample_rate)
            self.examples = paths[0:num_examples]
        else:
            self.examples = paths

    def __getitem__(self, item):
        path, slice, fname = self.examples[item]
        img_path = os.path.join(path, str(slice)+'.mat')
        img_path = img_path.replace('fastMRI_brain_DICOM', 'fastMRI_brain_DICOM_mat')
        img = loadmat(img_path)['img']

        kspace = fft2(img)
        kspace = center_crop(kspace, self.crop_size)
        maskedkspace = kspace * self.mask

        maskedkspace = to_tensor(maskedkspace)

        subsample = complex_abs(ifft2c(maskedkspace))

        kspace = to_tensor(kspace)
        target = complex_abs(ifft2c(kspace))

        subsample, mean, std = normalize_instance(subsample, eps=1e-11)
        target = normalize(target, mean, std, eps=1e-11)

        maskedkspace = maskedkspace.float()
        target = target.float()

        return maskedkspace, self.mask, target, mean, std, fname, slice

    def __len__(self):
        return len(self.examples)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]

def build_fastmri_dataset(args, mode='train'):
    mask = os.path.join(args.TRANSFORMS.DICOM_MASK_DIR, args.TRANSFORMS.MASK_FILE[0])
    if mode == 'train':
        data_list = os.path.join(args.DATASET.ROOT[0], 'train.txt')
    elif mode == 'val':
        data_list = os.path.join(args.DATASET.ROOT[0], 'valid.txt')

    return FastMRIDicom(data_list, mask, mode=mode, sample_rate=args.DATASET.SAMPLE_RATE[0])

