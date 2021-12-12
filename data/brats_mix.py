import os
import torch
import numpy as np
import nibabel as nib
import random

from .transforms import create_mask_for_mask_type, normalize, normalize_instance
from torch.utils.data import Dataset
from scipy.io import loadmat

def fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))

def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data

def load_nii(niipath):
    image = np.array(nib_load(niipath), dtype='float32', order='C')
    mask = image > 0
    t = image[mask]
    image[mask] -= t.mean()
    image[mask] /= t.std()
    return image

class BraTS(Dataset):
    def __init__(self, list_file, mask, root='', mode='train', sample_rate = 1):
        paths = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                path = os.path.join(root, line, name + '_')
                for slice in range(40, 120):
                    paths.append((path + 't1.nii.gz', slice, name + '_t1'))
                    paths.append((path + 't2.nii.gz', slice, name + '_t2'))
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

        img = load_nii(path)[..., slice].transpose(1, 0)
        kspace = fft2(img)
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


def build_brats_dataset(args, mode='train'):
    mask = os.path.join(args.TRANSFORMS.DICOM_MASK_DIR, args.TRANSFORMS.MASK_FILE[1])
    if mode == 'train':
        train_root = os.path.join(args.DATASET.ROOT[1], 'train')
        train_list = os.path.join(args.DATASET.ROOT[1], 'train', 'train.txt')
        return BraTS(train_list, mask, train_root, mode, args.DATASET.SAMPLE_RATE[1])
    elif mode == 'val':
        val_root = os.path.join(args.DATASET.ROOT[1], 'val')
        val_list = os.path.join(args.DATASET.ROOT[1], 'val', 'valid.txt')
        return BraTS(val_list, mask, val_root, mode, args.DATASET.SAMPLE_RATE[1])



