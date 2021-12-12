import csv
import os

import logging
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import pathlib

# import cv2
import scipy.io as sio
from scipy.io import loadmat,savemat
from os import listdir, path
from os.path import splitext
from types import SimpleNamespace

import h5py
from .math import ifft2c, fft2c, complex_abs
from .transforms import build_transforms, normalize_instance
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
# from BraTS import BraTS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def imshow(img, title=""):
    """ Show image as grayscale. """
    if img.dtype == np.complex64 or img.dtype == np.complex128:
        print('img is complex! Take absolute value.')
        img = np.abs(img)

    plt.figure()
    # plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.imsave( '{}.png'.format(title), img, cmap='gray')

def norm(data, eps=1e-11):
    data = (data - data.min()) / (data.max() - data.min() + eps)
    return data

def fetch_dir(key, data_config_file=pathlib.Path("fastmri_dirs.yaml")):
    """
    Data directory fetcher.

    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.

    Args:
        key (str): key to retrieve path from data_config_file.
        data_config_file (pathlib.Path,
            default=pathlib.Path("fastmri_dirs.yaml")): Default path config
            file.

    Returns:
        pathlib.Path: The path to the specified directory.
    """
    if not data_config_file.is_file():
        default_config = dict(
            knee_path="/home/jc3/Data/",
            brain_path="/home/jc3/Data/",
        )
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        raise ValueError(f"Please populate {data_config_file} with directory paths.")

    with open(data_config_file, "r") as f:
        data_dir = yaml.safe_load(f)[key]

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Path {data_dir} from {data_config_file} does not exist.")

    return data_dir


def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

#-----------------------------------------------------------------------------------------------------------------------------

class IXIdataset(Dataset):
    def __init__(
            self, 
            data_dir, 
            transforms,
            args, 
            challenge,
            sample_rate=1, 
            mode='train',
            pattern='T2',
            client_name='IXI',  # IXI_zi_1
            client_num=2,
    ):
        self.transform = transforms
        self.pattern = pattern
        self.img_size = args.img_size
        self.file_names = []
        # self.final_file_names = []
        self.examples = []

        if pattern == 'T1':
            self.data_list = [data_dir]
        elif pattern == 'T2':
            self.data_list = [data_dir.replace('/IXI/', '/IXI_T2/')]
        elif pattern == 'T1+T2':
            self.data_list = [data_dir, data_dir.replace('/IXI/', '/IXI_T2/')]

        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        metadata = {
            'acquisition': pattern,
            'encoding_size': (256, 256, 1),
            'max': 0,
            'norm': 0,
            'padding_left': 0,
            'padding_right': 0,
            'patient_id':'0',
            'recon_size': (256, 256, 1),
        }

        #make an image id's list
        for dataset in self.data_list:
            self.file_names += [(dataset, splitext(file)[0]) for file in listdir(dataset)
                        if not file.startswith('.')]

        for dataset, file_name in self.file_names:
            full_file_path = path.join(dataset, file_name+'.hdf5')
                
            for slice_id in range(20, 120, 1):
                self.examples.append((full_file_path, slice_id, metadata))

        # split to subdatasets
        if client_name.find('zi_')>=0:
            division_num = int(client_name.split('zi_')[-1])
            divided_examples = []

            for i in range(0, len(self.examples), len(self.examples)//client_num):
                if i == range(0, len(self.examples), len(self.examples)//client_num)[-1] and \
                                len(self.examples) % (len(self.examples)//client_num) != 0:
                    new_list = self.examples[i:]
                    divided_examples[-1] += new_list
                else:
                    new_list = self.examples[i: i+len(self.examples)//client_num]
                    divided_examples.append(new_list)

            self.examples = divided_examples[division_num]
        
        if sample_rate < 1:
            if mode == 'train':
                random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, slice_id):

        fname_nii, slice_idx, metadata =self.examples[slice_id]

        slice_path = self.h5path2matpath(fname_nii, slice_idx)
        image = loadmat(slice_path)['img']  # spatial  (556,640)

        mask = None
        attrs = metadata

        image = np.rot90(image)
        
        kspace = self.fft2c(image).astype(np.complex64)
        target = image.astype(np.float32)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname_nii, slice_idx)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname_nii, slice_idx)

        return sample

    def fft2c(self, img):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

    def h5path2matpath(self, fname, slice_id):
        filename, _ = fname.split('.')
        mat_dir = path.join(filename + '-{:03d}.mat'.format(slice_id))
        full_file_path = mat_dir.replace('/h5/', '/mat/')

        return full_file_path

class LianYingdataset(Dataset):
    def __init__(
        self, 
        data_dir,
        transforms,
        args,
        challenge,
        sample_rate=1,
        mode='train', 
        pattern='T2'
        ):
        
        self.transform = transforms
        self.data_dir = data_dir
        self.img_size = args.img_size
        self.examples = []

        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        #make an image id's list
        if mode == 'train':
            f = open(path.join(str(data_dir),'lianying_train.txt'),'r')#_little
        elif mode == 'val':
            f = open(path.join(str(data_dir),'lianying_val.txt'),'r')#_little
        elif mode == 'test':
            f = open(path.join(str(data_dir),'lianying_test.txt'),'r')#_little
        else: 
            raise ValueError("No mode like this, please choose one in ['train', 'val', 'test'].")
        
        file_names = f.readlines()

        metadata = {
            'acquisition': pattern,
            'encoding_size': (640, 556, 1),
            'max': 0,
            'norm': 0,
            'padding_left': 0,
            'padding_right': 0,
            'patient_id':'0',
            'recon_size': (320, 320, 1),
        }

        if not pattern == 'T1+T2':

            if pattern == 'T1':
                idx = 0
            elif pattern == 'T2':
                idx = 1

            for file_name in file_names:
                splits = file_name.split()  # 分离空格
                for slice_id in range(args.slice_range[0], args.slice_range[1]+1):  # 0:19==20
                    self.examples.append((splits[idx], slice_id, metadata))  # 获取T1/T2的slice列表
        else:
            for file_name in file_names:
                splits = file_name.split()
                for slice_id in range(args.slice_range[0], args.slice_range[1]+1):  # 0:19==20
                    self.examples.append((splits[0], slice_id, metadata))  # 获取T1的slice列表
                    self.examples.append((splits[1], slice_id, metadata))  # 获取T2的slice列表

        if mode == 'train':
            logging.info(f'Creating training dataset with {len(self.examples)} examples')
        elif mode == 'val':
            logging.info(f'Creating validation dataset with {len(self.examples)} examples')
        elif mode == 'test':
            logging.info(f'Creating test dataset with {len(self.examples)} examples')
        
        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, slice_id):

        fname_nii, slice_idx, metadata =self.examples[slice_id]

        slice_path = self.niipath2matpath(fname_nii, slice_idx)
        image = loadmat(slice_path)['img']  # spatial  (556,640)

        mask = None
        attrs = metadata

        image = np.rot90(image)
        
        kspace = self.fft2c(image).astype(np.complex64)
        target = image.astype(np.float32)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname_nii, slice_idx)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname_nii, slice_idx)

        return sample

    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[1] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size)/2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx

    def fft2c(self, img):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

    def niipath2matpath(self, T1,slice_id):
        filedir,filename = path.split(T1)
        filedir,_ = path.split(filedir)
        mat_dir = path.join(filedir,'mat_320')
        basename, ext = path.splitext(filename)
        base_name = basename[:-1]
        file_name = '%s-%03d.mat'%(base_name,slice_id)
        T1_file_path = path.join(mat_dir,file_name)
        return T1_file_path

    def center_crop(self, data, shape):
        assert 0 < shape[0] <= data.shape[-2], 'Error: shape: {}, data.shape: {}'.format(shape, data.shape)#556...556
        assert 0 < shape[1] <= data.shape[-1]#640...640
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to]


class JiangSudataset(Dataset):
    def __init__(
        self, 
        data_dir,
        transforms,
        args,
        challenge,
        sample_rate=1,
        mode='train', 
        pattern='T2'
        ):
        
        self.transform = transforms
        self.data_dir = data_dir
        self.img_size = args.img_size
        self.examples = []

        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        #make an image id's list
        if mode == 'train':
            f = open(path.join(str(data_dir),'jiangsu_train.txt'),'r')#_little
        elif mode == 'val':
            f = open(path.join(str(data_dir),'jiangsu_val.txt'),'r')#_little
        elif mode == 'test':
            f = open(path.join(str(data_dir),'jiangsu_test.txt'),'r')#_little
        else: 
            raise ValueError("No mode like this, please choose one in ['train', 'val', 'test'].")
        
        file_names = f.readlines()

        metadata = {
            'acquisition': pattern,
            'encoding_size': (640, 556, 1),
            'max': 0,
            'norm': 0,
            'padding_left': 0,
            'padding_right': 0,
            'patient_id':'0',
            'recon_size': (320, 320, 1),
        }

        if not pattern == 'T1+T2':

            if pattern == 'T1':
                idx = 0
            elif pattern == 'T2':
                idx = 1

            for file_name in file_names:
                splits = file_name.split()  # 分离空格
                # list_pattern.append(splits[idx])
                for slice_id in range(args.slice_range[0], args.slice_range[1]+1):  # 0:19==20
                    self.examples.append((splits[idx], slice_id, metadata))  # 获取T1/T2的slice列表
        else:
            for file_name in file_names:
                splits = file_name.split()
                for slice_id in range(args.slice_range[0], args.slice_range[1]+1):  # 0:19==20
                    self.examples.append((splits[0], slice_id, metadata))  # 获取T1的slice列表
                    self.examples.append((splits[1], slice_id, metadata))  # 获取T2的slice列表
        
        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, slice_id):

        fname_nii, slice_idx, metadata =self.examples[slice_id]

        slice_path = self.niipath2matpath(fname_nii, slice_idx)
        image = loadmat(slice_path)['img']  # spatial  (556,640)

        mask = None
        attrs = metadata

        image = np.rot90(image)
        
        kspace = self.fft2c(image).astype(np.complex64)
        target = image.astype(np.float32)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname_nii, slice_idx)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname_nii, slice_idx)

        return sample

    def crop_toshape(self, kspace_cplx):
        if kspace_cplx.shape[1] == self.img_size:
            return kspace_cplx
        if kspace_cplx.shape[0] % 2 == 1:
            kspace_cplx = kspace_cplx[:-1, :-1]
        crop = int((kspace_cplx.shape[0] - self.img_size)/2)
        kspace_cplx = kspace_cplx[crop:-crop, crop:-crop]
        return kspace_cplx

    def fft2c(self, img):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img)))

    def ifft2c(self, img):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))

    def fft2(self, img):
        return np.fft.fftshift(np.fft.fft2(img))

    def ifft2(self, kspace_cplx):
        return np.absolute(np.fft.ifft2(kspace_cplx))

    def niipath2matpath(self, T1,slice_id):
        filedir,filename = path.split(T1)
        filedir,_ = path.split(filedir)
        mat_dir = path.join(filedir,'mat_320')
        basename, ext = path.splitext(filename)
        base_name = basename[:-1]
        file_name = '%s-%03d.mat'%(base_name,slice_id)
        T1_file_path = path.join(mat_dir,file_name)
        return T1_file_path

    def center_crop(self, data, shape):
        assert 0 < shape[0] <= data.shape[-2], 'Error: shape: {}, data.shape: {}'.format(shape, data.shape)#556...556
        assert 0 < shape[1] <= data.shape[-1]#640...640
        w_from = (data.shape[-2] - shape[0]) // 2
        h_from = (data.shape[-1] - shape[1]) // 2
        w_to = w_from + shape[0]
        h_to = h_from + shape[1]
        return data[..., w_from:w_to, h_from:h_to]


class SliceDataset(Dataset):
    def __init__(
            self,
            root,
            transform,
            challenge,
            sample_rate=1,
            mode='train',
            pattern='pd',
    ):

        # challenge
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        # transform
        self.transform = transform

        self.examples = []

        self.cur_path = root
        if pattern == 'mix0' or pattern == 'mix1':
            self.csv_file = os.path.join(self.cur_path, "singlecoil_" + mode + "_mixsplit_less.csv")
        elif pattern == 'pd' or pattern == 'pdfs' or pattern == 'pd+pdfs':
            self.csv_file = os.path.join(self.cur_path, "singlecoil_" + mode + "_split_less.csv")
        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                if not pattern == 'pd+pdfs':
                    if pattern == 'pd' or pattern == 'mix0':
                        idx = 0
                    elif pattern == 'pdfs' or pattern == 'mix1':
                        idx = 1

                    metadata, num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[idx] + '.h5'))

                    for slice_id in range(num_slices):
                        self.examples.append((os.path.join(self.cur_path, row[idx] + '.h5'), slice_id, metadata))
                
                else:
                    pd_metadata, pd_num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[0] + '.h5'))

                    pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(self.cur_path, row[1] + '.h5'))

                    for slice_id in range(min(pd_num_slices, pdfs_num_slices)):
                        self.examples.append(
                            (os.path.join(self.cur_path, row[0] + '.h5'), slice_id, pd_metadata))
                        self.examples.append(
                            (os.path.join(self.cur_path, row[1] + '.h5'), slice_id, pdfs_metadata))

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)

            self.examples = self.examples[0:num_examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        # 读取pd
        fname, slice, metadata = self.examples[i]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][slice]

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][slice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)

            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname, slice)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname, slice)

        return sample  # image, target, mean, std, fname, slice_num

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices

#-----------------------------------------------------------------------------------------------------------------------------

def create_datasets(args, mode='train', sample_rate=1, client_name='fastMRI', pattern='pd', client_num=2):
    assert mode in ['train', 'val', 'test'], 'unknown mode'
    transforms = build_transforms(args, mode, client_name=client_name)


    if client_name == 'JiangSu':
        # path_config = os.path.join(args.DATASET.ROOT, 'JiangSu', 'config.yaml')
        path_config = os.path.join('./config/config_{}.yaml'.format(client_name.lower()))
        with open(path_config) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            ixi_args = SimpleNamespace(**data)
        return JiangSudataset(os.path.join(args.DATASET.ROOT[2], 'JiangSu'), transforms, ixi_args, args.DATASET.CHALLENGE,
                            sample_rate=sample_rate, mode=mode, pattern=pattern)
    
    elif client_name == 'lianying':
        path_config = os.path.join('./config/config_{}.yaml'.format(client_name.lower()))
        with open(path_config) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            ixi_args = SimpleNamespace(**data)
        return LianYingdataset(os.path.join(args.DATASET.ROOT[3], 'lianying'), transforms, ixi_args, args.DATASET.CHALLENGE,
                            sample_rate=sample_rate, mode=mode, pattern=pattern)
    
    elif client_name == 'IXI':
        path_config = os.path.join('./config/config_{}.yaml'.format(client_name.lower()))
        with open(path_config) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            ixi_args = SimpleNamespace(**data)
        return IXIdataset(os.path.join(args.DATASET.ROOT, ixi_args.dataset, 'h5', mode), transforms, ixi_args, args.DATASET.CHALLENGE,
                            sample_rate=sample_rate, mode=mode, pattern=pattern)
    
    elif client_name.find('IXI_zi_') >= 0:
        path_config = os.path.join('./config/config_{}.yaml'.format(client_name.split('_')[0].lower()))
        with open(path_config) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            ixi_args = SimpleNamespace(**data)
        return IXIdataset(os.path.join(args.DATASET.ROOT, ixi_args.dataset, 'h5', mode), transforms, ixi_args, args.DATASET.CHALLENGE,
                            sample_rate=sample_rate, mode=mode, pattern=pattern, client_name=client_name, client_num=client_num)
    else:
        return SliceDataset(os.path.join(args.DATASET.ROOT, client_name,'singlecoil_' + mode), transforms, args.DATASET.CHALLENGE,
                            sample_rate=sample_rate, mode=mode, pattern=pattern)
