import torch
import time
import os
import datetime
import random
import copy
import numpy as np
import argparse

from models.unet_module import *
from models.loss import Criterion
from data.fastmri import create_datasets
from torch.utils.data import DataLoader, DistributedSampler
from config import build_config
from pathlib import Path
from engine import train_one_epoch, train_one_epoch, evaluate, train_one_epoch_ours
from util.misc import init_distributed_mode, get_rank
from util.utils import *
from data import build_different_dataloader
from models.unet import KIKNET

################# Key Function ########################
def communication(args, server_model1, server_model2, models, client_weights):
    prev_models = [copy.deepcopy(model) for model in models]
    with torch.no_grad():
        for key in server_model1.state_dict().keys():
            temp = torch.zeros_like(server_model1.state_dict()[key])
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()['unet1.'+key]
            server_model1.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                models[client_idx].state_dict()['unet1.'+key].data.copy_(server_model1.state_dict()[key])

        for key in server_model2.state_dict().keys():
            temp = torch.zeros_like(server_model2.state_dict()[key])
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()['unet2.'+key]
            server_model2.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                models[client_idx].state_dict()['unet2.'+key].data.copy_(server_model2.state_dict()[key])
    return models, prev_models


def create_all_model(args, share='only_encoder', client_num=2):
    device = torch.device(args.SOLVER.DEVICE)
    server_model1 = Encoder(in_chans=2)
    server_model2 = Encoder(in_chans=1)
    model_0 = KIKNET(in_chans=2, out_chans=1)
    models = [copy.deepcopy(model_0).to(device) for idx in range(len(args.DATASET.CLIENTS))]
    for model in models:
        for key in server_model1.state_dict().keys():
            model.state_dict()['unet1.'+key].data.copy_(server_model1.state_dict()[key])
    for model in models:
        for key in server_model2.state_dict().keys():
            model.state_dict()['unet2.'+key].data.copy_(server_model2.state_dict()[key])

    prev_models = [copy.deepcopy(model) for model in models]
    return server_model1, server_model2, models, prev_models

def prlog(i, args=cfg):
    print(i)
    # with open('./logs/{}/log.txt'.format(args.FL.MODEL_NAME), 'a+') as f:
    #     f.write(i)
    #     f.write('\n')


def main(args):
    # build criterion and model first
    init_distributed_mode(args)
    args.OUTPUTDIR = os.path.join(args.OUTPUTDIR, args.FL.MODEL_NAME)
    args.LOGDIR = os.path.join(args.LOGDIR, args.FL.MODEL_NAME)
    print(args.LOGDIR)
    os.makedirs(args.OUTPUTDIR, exist_ok=True)
    os.makedirs(args.LOGDIR, exist_ok=True)
    prlog('\n\n\n')
    prlog('New job assigned {}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M')))
    prlog('\nconfig:\n{}\n'.format(args))
    for arg in vars(args):
        prlog('{}: {}\n'.format(arg, getattr(args, arg)))

    server_model1, server_model2, models, prev_models = create_all_model(args, args.FL.SHARE_WAY, args.FL.CLIENTS_NUM)
    criterion = Criterion(args)

    start_epoch = 0

    seed = args.SEED + get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.SOLVER.DEVICE)

    client_weights = [1 / args.FL.CLIENTS_NUM for i in range(args.FL.CLIENTS_NUM)]

    assessment_dict = [
        {'loss_train_epoch': [], 'loss_val_epoch': [], 'nmse_epoch': [], 'psnr_epoch': [], 'ssim_epoch': []} for i in
        range(args.FL.CLIENTS_NUM)]

    server_model1.to(device)
    server_model2.to(device)
    for model in models:
        model.to(device)
    criterion.to(device)

    # show params for server/clients
    n_parameters = sum(p.numel() for p in server_model1.parameters() if p.requires_grad)
    prlog('Volume of SERVER model params:    {:.2f} M'.format(n_parameters / 1024 / 1024))
    for idx, model in enumerate(models):
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        prlog('Volume of CLIENT {:<8s} params: {:.2f} M'.format(args.DATASET.CLIENTS[idx], n_parameters / 1024 / 1024))

    # build optimizer
    optimizers = [torch.optim.RMSprop(params=models[idx].parameters(), lr=args.SOLVER.LR[idx]) for idx in
                  range(args.FL.CLIENTS_NUM)]
    lr_schedulers = [torch.optim.lr_scheduler.StepLR(optimizers[idx], args.SOLVER.LR_DROP, args.SOLVER.LR_GAMMA) for idx
                     in range(args.FL.CLIENTS_NUM)]

    dataloader_train, _ = build_different_dataloader(args, mode='train')
    for dataloader in dataloader_train:
        print(len(dataloader))
    dataloader_val, _ = build_different_dataloader(args, mode='val')

    if len(args.DATASET.CLIENTS) == 2 and args.FL.DATAMIX:
        data_name = ['mix0', 'mix1']
    else:  # NOTE
        data_name = args.DATASET.CLIENTS

    if args.RESUME != '':
        checkpoint = torch.load(args.RESUME)
        checkpoint = checkpoint['server_model']
        checkpoint = {key.replace("module.", ""): val for key, val in checkpoint.items()}
        prlog('resume from %s' % args.RESUME)
        server_model1.load_state_dict(checkpoint['server_model'], strict=True)
        for idx, client_name in enumerate(args.DATASET.CLIENTS):
            models[idx].load_state_dict(checkpoint['model_{}'.format(client_name)])
            optimizers[idx].load_state_dict(checkpoint['optimizer_{}'.format(client_name)])
            lr_schedulers[idx].load_state_dict(checkpoint['lr_scheduler_{}'.format((client_name))])
        start_epoch = checkpoint['epoch'] + 1

    start_time = time.time()

    best_status = [{'NMSE': 10000000, 'PSNR': 0, 'SSIM': 0} for i in range(args.FL.CLIENTS_NUM)]
    best_checkpoint = [{} for i in range(args.FL.CLIENTS_NUM)]

    for epoch in range(start_epoch, args.TRAIN.EPOCHS):
        prlog('------------------ Epoch {:<3d}---------------------'.format(epoch + 1))
        for client_idx in range(args.FL.CLIENTS_NUM):

            for _ in range(args.TRAIN.SMALL_EPOCHS):
                if epoch > 0:
                    train_status = train_one_epoch_ours(args,
                                                        models[client_idx], server_model1, server_model2, prev_models, criterion,
                                                        dataloader_train[client_idx],
                                                        optimizers[client_idx], epoch, args.SOLVER.PRINT_FREQ, device)
                    prlog(' {:<11s}| Train Loss: {:.4f} | C_Loss: {:.4f} | P_Loss: {:.4f}'.format(data_name[client_idx],
                                                                                                  train_status['loss'],
                                                                                                  train_status[
                                                                                                      'c_loss'],
                                                                                                  train_status[
                                                                                                      'p_loss']))
                else:
                    train_status = train_one_epoch(args,
                                                   models[client_idx], criterion, dataloader_train[client_idx],
                                                   optimizers[client_idx], epoch, args.SOLVER.PRINT_FREQ, device)
                    prlog(' {:<11s}| Train Loss: {:.4f}'.format(data_name[client_idx], train_status['loss']))

            lr_schedulers[client_idx].step()

        # aggregation
        models, prev_models = communication(args, server_model1,server_model2, models, client_weights)

        for client_idx in range(args.FL.CLIENTS_NUM):


            model, val_loader = models[client_idx], dataloader_val[client_idx]
            eval_status = evaluate(args, model, criterion, val_loader, device, data_name[client_idx])
            if eval_status['PSNR'] > best_status[client_idx]['PSNR']:
                best_status[client_idx] = eval_status
                best_checkpoint[client_idx] = {
                    'server_model': server_model1.state_dict(),
                    'model_{}'.format(client_idx): models[client_idx].state_dict(),
                    'optimizer': optimizers[client_idx].state_dict(),
                    'lr_scheduler': lr_schedulers[client_idx].state_dict(),
                    'epoch': epoch,
                    'args': args,
                }

            # save model
            if args.OUTPUTDIR:
                Path(args.OUTPUTDIR).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(args.OUTPUTDIR, f'checkpoint-epoch_{(epoch + 1):04}.pth')
                checkpoint = {'server_model': server_model1.state_dict()}
                for idx, client_name in enumerate(args.DATASET.CLIENTS):
                    checkpoint.update({
                        'model_{}'.format(client_name): models[idx].state_dict(),
                        'optimizer_{}'.format(client_name): optimizers[idx].state_dict(),
                        'lr_scheduler_{}'.format((client_name)): lr_schedulers[idx].state_dict()
                    })

                checkpoint.update({
                    'epoch': epoch,
                    'args': args,
                })

                torch.save(checkpoint, checkpoint_path)

    for idx, client_name in enumerate(args.DATASET.CLIENTS):
        prlog('The best epoch for CLIENT {:<8s} is {}'.format(client_name, best_checkpoint[idx]['epoch'] + 1))
        prlog("Results ----------")
        prlog("NMSE: {:.4}".format(best_status[idx]['NMSE']))
        prlog("PSNR: {:.4}".format(best_status[idx]['PSNR']))
        prlog("SSIM: {:.4}".format(best_status[idx]['SSIM']))
        prlog("------------------")

    if args.OUTPUTDIR:
        for idx, client_name in enumerate(args.DATASET.CLIENTS):
            checkpoint_path = os.path.join(args.OUTPUTDIR, 'client_{}_best.pth'.format(client_name))
            torch.save(best_checkpoint[idx], checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    prlog('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="a unit Cross Multi modity transformer")
    parser.add_argument(
        "--config", default="different_dataset_client", help="choose a experiment to do")
    args = parser.parse_args()

    cfg = build_config(args.config)
    main(cfg)