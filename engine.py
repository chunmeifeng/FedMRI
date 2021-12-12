import time
import hashlib
from typing import Iterable
import util.misc as utils
import datetime
import numpy as np

from util.metric import nmse, psnr, ssim, AverageMeter
from collections import defaultdict
from config import cfg

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./log/tensorboard')


def norm(data, dtype='max', eps=1e-11):
    data = (data - data.min()) / (data.max() - data.min() + eps)
    return data


def prlog(i, args=cfg):
    print(i)
    with open('./logs/{}/log.txt'.format(args.FL.MODEL_NAME), 'a+') as f:
        f.write(i)
        f.write('\n')

def train_one_epoch_ours(args, model: torch.nn.Module, server_model: torch.nn.Module,
                           prev_models, criterion: torch.nn.Module,
                           data_loader: Iterable, optimizer: torch.optim.Optimizer,
                           epoch: int, print_freq: int, device: str):
    model.train()
    loss_all = 0
    c_loss = 0
    p_loss = 0

    server_model.eval()
    for prev in prev_models:
        prev.eval()

    for i, data in enumerate(data_loader):
        image, target, mean, std, fname, slice_num = data  # NOTE

        image = image.unsqueeze(1)  # (8,1,320,320)
        target = target.unsqueeze(1)

        image = image.to(device)
        target = target.to(device)

        outputs = model(image)

        loss = criterion(outputs, target)

        if i > 0:
            posi = torch.tensor(0., device=device)
            nega = torch.tensor(0., device=device)
            for name, param in server_model.named_parameters():
                curr_params = dict(model.named_parameters())[name]
                posi += torch.norm(curr_params - param, p=1)

            for prev_model in prev_models:
                for name, _ in server_model.named_parameters():
                    prev_param = dict(prev_model.named_parameters())[name]
                    curr_params = dict(model.named_parameters())[name]
                    nega += torch.norm(curr_params - prev_param, p=1)

            w_diff = posi / (nega + 1e-14)

            total_loss = args.lam * loss['loss'] + args.beta * w_diff
            loss_all += args.lam * loss['loss'].item() + args.beta * w_diff.item()
            p_loss += args.beta * w_diff.item()

        else:
            total_loss = loss['loss']
            loss_all += loss['loss'].item()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        c_loss += args.lam * loss['loss'].item()

    c_loss_avg = c_loss / len(data_loader)
    p_loss_avg = p_loss / len(data_loader)
    loss_avg = loss_all / len(data_loader)
    global_step = int(epoch * len(data_loader) + len(data_loader))

    return {"loss": loss_avg, "global_step": global_step, 'c_loss': c_loss_avg, 'p_loss': p_loss_avg}


def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                           data_loader: Iterable, optimizer: torch.optim.Optimizer,
                           epoch: int, print_freq: int, device: str):
    model.train()
    loss_all = 0
    for _, data in enumerate(data_loader):
        image, target, mean, std, fname, slice_num = data  # NOTE

        image = image.unsqueeze(1)  # (8,1,320,320)
        target = target.unsqueeze(1)

        image = image.to(device)
        target = target.to(device)

        outputs = model(image)

        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        loss_all += loss['loss'].item()

    loss_avg = loss_all / len(data_loader)
    global_step = int(epoch * len(data_loader) + len(data_loader))

    return {"loss": loss_avg, "global_step": global_step}


@torch.no_grad()
def evaluate(args, model, criterion, data_loader, device, data_name):
    model.eval()
    criterion.eval()
    criterion.to(device)

    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    # _nmse_meter = AverageMeter()
    # _psnr_meter = AverageMeter()
    # _ssim_meter = AverageMeter()

    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)

    start_time = time.time()

    loss_all = 0

    for data in data_loader:
        image, target, mean, std, fname, slice_num = data  # torch.float32
        image = image.unsqueeze(1)

        mean = mean.unsqueeze(1).unsqueeze(2)  # (8,1,1)
        std = std.unsqueeze(1).unsqueeze(2)
        mean = mean.to(device)
        std = std.to(device)

        image = image.to(device)
        target = target.to(device)

        b = image.shape[0]

        outputs = model(image)
        outputs = outputs.squeeze(1)

        outputs = outputs * std + mean
        target = target * std + mean

        loss = criterion(outputs, target)
        loss_all += loss['loss'].item()

        # our_nmse = nmse(target.cpu().numpy(), outputs.cpu().numpy())
        # our_psnr = psnr(norm(target.cpu().numpy()), norm(outputs.cpu().numpy()))
        # our_ssim = ssim(target.cpu().numpy(), outputs.cpu().numpy())

        # nmse_meter.update(our_nmse, b)
        # psnr_meter.update(our_psnr, b)
        # ssim_meter.update(our_ssim, b)

        for i, f in enumerate(fname):
            output_dic[f][slice_num[i]] = outputs[i]
            target_dic[f][slice_num[i]] = target[i]

    for name in output_dic.keys():
        f_output = torch.stack([v for _, v in output_dic[name].items()])  # (34,320,320)
        f_target = torch.stack([v for _, v in target_dic[name].items()])  # (34,320,320)
        # our_nmse = nmse(norm(f_target.cpu().numpy()), norm(f_output.cpu().numpy()))
        # our_psnr = psnr(norm(f_target.cpu().numpy()), norm(f_output.cpu().numpy()))
        # our_ssim = ssim(norm(f_target.cpu().numpy()), norm(f_output.cpu().numpy()))
        our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

        nmse_meter.update(our_nmse, 1)
        psnr_meter.update(our_psnr, 1)
        ssim_meter.update(our_ssim, 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    loss_avg = loss_all / len(data_loader)

    prlog(' {:<11s}|   Val Loss: {:.4f} Evaluate time {} NMSE: {:.4f} PSNR: {:.4f} SSIM: {:.4f}'.format(data_name,
                                                                                                        loss_avg,
                                                                                                        total_time_str,
                                                                                                        nmse_meter.avg,
                                                                                                        psnr_meter.avg,
                                                                                                        ssim_meter.avg))

    return {'loss': loss_avg, 'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM': ssim_meter.avg}

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


@torch.no_grad()
def distributed_evaluate(args, model,  data_loader, device, data_name, dataset_len):
    model.eval()

    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    start_time = time.time()

    output_list = []
    target_list = []
    id_list = []
    slice_list = []

    for data in data_loader:
        image, target, mean, std, fname, slice_num = data  # torch.float32
        image = image.unsqueeze(1)

        mean = mean.unsqueeze(1).unsqueeze(2)  # (8,1,1)
        std = std.unsqueeze(1).unsqueeze(2)
        mean = mean.to(device)
        std = std.to(device)

        image = image.to(device)
        target = target.to(device)
        outputs = model(image)

        outputs = outputs.squeeze(1)
        outputs = outputs * std + mean
        target = target * std + mean

        fid = torch.zeros(len(fname), dtype=torch.long, device=outputs.device)
        for i, fn in enumerate(fname):
            fid[i] = (
                int(hashlib.sha256(fn.encode("utf-8")).hexdigest(), 16) % 10 ** 12
            )

        output_list.append(outputs)
        target_list.append(target)
        id_list.append(fid)
        slice_list.append(slice_num)

    final_id = distributed_concat(torch.cat((id_list), dim=0), dataset_len)
    final_output = distributed_concat(torch.cat((output_list), dim=0), dataset_len)
    final_target = distributed_concat(torch.cat((target_list), dim=0), dataset_len)
    final_slice = distributed_concat(torch.cat((slice_list), dim=0), dataset_len)

    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)

    final_id = final_id.cpu().numpy()

    for i, f in enumerate(final_id):
        output_dic[f][final_slice[i]] = final_output[i]
        target_dic[f][final_slice[i]] = final_target[i]

    for name in output_dic.keys():
        f_output = torch.stack([v for _, v in output_dic[name].items()])
        f_target = torch.stack([v for _, v in target_dic[name].items()])
        our_nmse = nmse(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_psnr = psnr(f_target.cpu().numpy(), f_output.cpu().numpy())
        our_ssim = ssim(f_target.cpu().numpy(), f_output.cpu().numpy())

        nmse_meter.update(our_nmse, 1)
        psnr_meter.update(our_psnr, 1)
        ssim_meter.update(our_ssim, 1)



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    prlog(' {:<11s}|    Evaluate time {} NMSE: {:.4f} PSNR: {:.4f} SSIM: {:.4f}'.format(data_name, total_time_str,
                                                                                                    nmse_meter.avg,
                                                                                                    psnr_meter.avg,
                                                                                                    ssim_meter.avg))

    return {'NMSE': nmse_meter.avg, 'PSNR': psnr_meter.avg, 'SSIM':ssim_meter.avg}