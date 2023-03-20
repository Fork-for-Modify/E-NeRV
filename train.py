import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import os
from os.path import join as opj, exists as ope
from model import model_dict
from datasets import dataset_dict
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, dataloader
from torch.utils.data.distributed import DistributedSampler
from engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
import utils.misc as utils
from logger import get_logger


def get_args_parse():
    parser = argparse.ArgumentParser('Dense NeRV', add_help=False)

    # runtime
    parser.add_argument('--gpus', type=str, default='none',
                        help='visible gpu ids')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--output_dir', default='./outputs', type=str,
                        help='path to save the log and other files')
    parser.add_argument('--exp_name', default='EXP/train', type=str,
                        help='experiment name')
    parser.add_argument('--port', default=29500, type=int, help='port number')

    # config
    parser.add_argument('--cfg_path', default='', type=str,
                        help='path to specific cfg yaml file path')
    parser.add_argument('--resume_path', default='', type=str,
                        help='resume ckp')
    parser.add_argument('--save_image', action='store_true',
                        default=False, help='save image in tensorboard when evaluation')
    return parser


def main(args):
    # info
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = opj(args.output_dir, args.exp_name, time_str)
    os.makedirs(exp_dir, exist_ok=True)
    img_out_dir = opj(exp_dir, 'img_out')
    os.makedirs(img_out_dir, exist_ok=True)

    # env init
    logger = get_logger(opj(exp_dir, 'runtime.log'))  # logger
    writer = SummaryWriter(os.path.join(exp_dir, 'tensorboard'))  # writer
    if args.gpus != "none":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    utils.init_distributed_mode(args)
    logger.info('git:\n {}\n'.format(utils.get_sha()))

    # get cfg yaml file
    cfg = utils.load_yaml_as_dict(args.cfg_path)

    # fix the seed
    seed = cfg['seed']
    # seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model definition
    device = torch.device(args.device)
    model = model_dict[cfg['model']['model_name']](cfg=cfg['model'])
    model.to(device)

    # load model weights
    if args.resume_path:
        logger.info("=> resuming checkpoint from '{}'".format(args.resume_path))
        checkpoint = torch.load(args.resume_path)
        state_dict = checkpoint['model']
        resume_epoch = checkpoint['epoch']
        model.load_state_dict(state_dict)
        logger.info("=> loaded checkpoint from '{}' (epoch {})".format(
            args.resume_path, resume_epoch))
        start_epoch = resume_epoch
    else:
        start_epoch = 0

    # dump the cfg yaml file in exp output dir
    cfg['resume_path'] = args.resume_path
    cfg['start_epoch'] = start_epoch
    utils.dump_cfg_yaml(cfg, exp_dir)
    logger.info(cfg)

    model_without_ddp = model
    # get model params
    if args.rank in [0, None]:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
        logger.info(f'{args}\n {model}\n Model Params: {params}M')

    # dataset definition
    img_transform = transforms.ToTensor()
    dataset_train = dataset_dict[cfg['dataset_type']](
        vid_dir=cfg['dataset_path'], transform=img_transform, train=True)  # img_size=cfg['img_size'],
    dataset_val = dataset_dict[cfg['dataset_type']](
        vid_dir=cfg['dataset_path'], transform=img_transform, train=False)  # img_size=cfg['img_size'],
    # follow nerv implementation on sampler and dataloader
    sampler_train = DistributedSampler(
        dataset_train) if args.distributed else None
    sampler_val = DistributedSampler(dataset_val) if args.distributed else None

    dataloader_train = DataLoader(
        dataset_train, batch_size=cfg['train_batchsize'], shuffle=(sampler_train is None), num_workers=cfg['workers'],
        pin_memory=True, sampler=sampler_train, drop_last=True, worker_init_fn=utils.worker_init_fn
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=cfg['val_batchsize'], shuffle=False, num_workers=cfg['workers'],
        pin_memory=True, sampler=sampler_val, drop_last=False, worker_init_fn=utils.worker_init_fn
    )

    datasize = len(dataset_train)

    # optimizer definition
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad],
            "lr": cfg['optim']['lr'],
        }
    ]

    optim_cfg = cfg['optim']
    if optim_cfg['optim_type'] == 'Adam':
        optimizer = optim.Adam(param_dicts, lr=optim_cfg['lr'], betas=(
            optim_cfg['beta1'], optim_cfg['beta2']))
    else:
        optimizer = None
    assert optimizer is not None, "No implementation of Optimizer!"

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # training
    logger.info('--- Start training ---')
    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [
        torch.tensor(0) for _ in range(4)]
    start_time = datetime.now()
    for epoch in range(start_epoch, cfg['epoch']):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, dataloader_train, optimizer, device, epoch, cfg, args, datasize, start_time, logger, writer)

        train_best_psnr = train_stats['train_psnr'][-1] if train_stats['train_psnr'][-1] > train_best_psnr else train_best_psnr
        train_best_msssim = train_stats['train_msssim'][-1] if train_stats['train_msssim'][-1] > train_best_msssim else train_best_msssim
        if args.rank in [0, None]:
            print_str = '==> Train: psnr_now: {:.2f}\t psnr_best: {:.2f}\t msssim_now: {:.4f}\t msssim_best: {:.4f}\n'.format(
                train_stats['train_psnr'][-1].item(), train_best_psnr.item(), train_stats['train_msssim'][-1].item(), train_best_msssim.item())
            logger.info(print_str)

        # save one per epoch
        checkpoint_path = opj(exp_dir, f'checkpoint_{epoch+1:03d}.pth')
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'config': cfg,
            'train_best_psnr': train_best_psnr,
            'train_best_msssim': train_best_msssim,
            'val_best_psnr': val_best_psnr,
            'val_best_msssim': val_best_msssim,
        }, checkpoint_path)
        logger.info(f'--- Model Saved: {checkpoint_path}')

        # delete previous ckp
        last_ckp = opj(exp_dir, f'checkpoint_{epoch:03d}.pth')
        if ope(last_ckp):
            os.remove(last_ckp)

        # evaluation
        if (epoch + 1) % cfg['eval_freq'] == 0 or epoch > cfg['epoch'] - 10:
            val_stats = evaluate(model, dataloader_val,
                                 device, cfg, args, epoch, logger,  args.save_image, img_out_dir)

            val_best_psnr = val_stats['val_psnr'][-1] if val_stats['val_psnr'][-1] > val_best_psnr else val_best_psnr
            val_best_msssim = val_stats['val_msssim'][-1] if val_stats['val_msssim'][-1] > val_best_msssim else val_best_msssim
            if args.rank in [0, None]:
                print_str = '==> Eval: psnr_now: {:.2f}\t psnr_best: {:.2f}\t msssim_now: {:.4f}\t msssim_best: {:.4f}\n'.format(
                    val_stats['val_psnr'][-1].item(), val_best_psnr.item(), val_stats['val_msssim'][-1].item(), val_best_msssim.item())
                logger.info(print_str)
        logger.info(
            f"--- Total Training Time: {str(datetime.now() - start_time)} ---\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'E-NeRV training and evaluation script', parents=[get_args_parse()])
    args = parser.parse_args()

    assert args.cfg_path is not None, 'Need a specific cfg path!'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
