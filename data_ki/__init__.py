import copy
import os
import torch
from .dicom_mix import FastMRIDicom_Split, build_fastmri_dataset
from .brats_mix import build_brats_dataset
from .fastmri import create_datasets
from torch.utils.data import DataLoader, DistributedSampler

def build_dataloader(args, mode='train'):
    assert len(args.TRANSFORMS.MASK_FILE) == args.FL.CLIENTS_NUM, 'please check the mask'
    mask = [os.path.join(args.TRANSFORMS.MASK_DIR, f) for f in args.TRANSFORMS.MASK_FILE]
    if mode == 'train':
        data_list = os.path.join(args.DATASET.ROOT, 'train.txt')
    elif mode == 'val':
        data_list = os.path.join(args.DATASET.ROOT, 'valid.txt')
    data_loader = []

    for i in range(args.FL.CLIENTS_NUM):
        dataset = FastMRIDicom_Split(data_list, mask, client_idx=i, client_num=args.FL.CLIENTS_NUM, mode=mode, sample_rate=args.DATASET.SAMPLE_RATE)
        if mode == 'train':
            data_loader.append(DataLoader(dataset, batch_size=args.SOLVER.BATCH_SIZE,
                                         num_workers=args.SOLVER.NUM_WORKERS, pin_memory=True,
                                         shuffle=True, drop_last=True))
        elif mode == 'val':
            data_loader.append(DataLoader(dataset, batch_size=args.SOLVER.BATCH_SIZE,  # no shuffle for val
                                         num_workers=args.SOLVER.NUM_WORKERS, pin_memory=True,
                                         shuffle=False))

    return data_loader

def build_different_dataloader(args, mode='train'):
    assert len(args.TRANSFORMS.MASK_FILE) == args.FL.CLIENTS_NUM, 'please check the mask'
    fastmri_dataset = build_fastmri_dataset(args, mode)
    brats_dataset = build_brats_dataset(args, mode)
    jiangsu_dataset = create_datasets(args, mode=mode, sample_rate=args.DATASET.SAMPLE_RATE[2], client_name='JiangSu', pattern=args.DATASET.PATTERN[2], client_num=4)
    lianying_dataset = create_datasets(args, mode=mode, sample_rate=args.DATASET.SAMPLE_RATE[3], client_name='lianying', pattern=args.DATASET.PATTERN[3], client_num=4)

    datasets = [fastmri_dataset, brats_dataset, jiangsu_dataset, lianying_dataset]



    data_loader = []
    dataset_len = []
    for dataset in datasets:
        dataset_len.append(len(dataset))
        if mode == 'train':
            if args.distributed:
                sampler = DistributedSampler(dataset)
            else:
                sampler = torch.utils.data.RandomSampler(dataset)

            data_loader.append(DataLoader(dataset, batch_size=args.SOLVER.BATCH_SIZE, sampler=sampler,
                                         num_workers=args.SOLVER.NUM_WORKERS, pin_memory=True, drop_last=True))
        elif mode == 'val':
            if args.distributed:
                sampler = DistributedSampler(dataset, shuffle=False)
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)
            data_loader.append(DataLoader(dataset, batch_size=args.SOLVER.BATCH_SIZE,  sampler=sampler,
                                         num_workers=args.SOLVER.NUM_WORKERS, pin_memory=True))
    return data_loader, dataset_len


