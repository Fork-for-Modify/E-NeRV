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
from engine import test
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
    parser.add_argument('--exp_name', default='EXP/test', type=str,
                        help='experiment name')
    parser.add_argument('--port', default=29500, type=int, help='port number')

    # config
    parser.add_argument('--cfg_path', default='', type=str,
                        help='path to specific cfg yaml file path')
    parser.add_argument('--ckp_path', default='', type=str,
                        help='checkpoint path')
    parser.add_argument('--eval', action='store_true',
                        default=False, help='performance evaluation')

    # input
    parser.add_argument('--frame_idx', default='', type=str,
                        help='a eval string to assign frame idx')
    parser.add_argument('--vid_path', default='', type=str,
                        help='a eval string to assign frame idx')
    return parser


def main(args):
    # info
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = opj(args.output_dir, args.exp_name, time_str)
    img_out_dir = opj(exp_dir, 'img_out')
    os.makedirs(img_out_dir, exist_ok=True)

    # env init
    logger = get_logger(opj(exp_dir, 'runtime.log'))  # logger
    if args.gpus != "none":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    utils.init_distributed_mode(args)
    logger.info('git:\n {}\n'.format(utils.get_sha()))

    # get cfg yaml file
    cfg = utils.load_yaml_as_dict(args.cfg_path)

    # fix the seed
    seed = cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # model definition
    device = torch.device(args.device)
    model = model_dict[cfg['model']['model_name']](cfg=cfg['model'])
    model.to(device)

    # calc model params
    if args.rank in [0, None]:
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
        logger.info(f'{args}\n {model}\n Model Params: {params}M')

    # load model weights
    logger.info("=> loading checkpoint from '{}'".format(args.ckp_path))
    checkpoint = torch.load(args.ckp_path)
    state_dict = checkpoint['model']
    resume_epoch = checkpoint['epoch']
    model.load_state_dict(state_dict)
    logger.info("=> loaded checkpoint from '{}' (epoch {})".format(
        args.ckp_path, resume_epoch))

    # dump the cfg yaml file in exp output dir
    cfg['ckp_path'] = args.ckp_path
    cfg['vid_path'] = args.vid_path
    cfg['frame_idx'] = args.frame_idx
    cfg['resume_epoch'] = resume_epoch
    utils.dump_cfg_yaml(cfg, exp_dir)
    logger.info(cfg)

    # dataset definition
    if args.frame_idx:
        frame_idx = eval(args.frame_idx)  # input frame indexes
    else:
        frame_idx = None

    img_transform = transforms.ToTensor()
    dataset_eval = dataset_dict[cfg['dataset_type']](
        frame_idx=frame_idx, vid_dir=args.vid_path, transform=img_transform, train=False)  #
    sampler_eval = DistributedSampler(
        dataset_eval) if args.distributed else None

    dataloader_test = DataLoader(
        dataset_eval, batch_size=cfg['val_batchsize'], shuffle=False, num_workers=cfg['workers'],
        pin_memory=True, sampler=sampler_eval, drop_last=False, worker_init_fn=utils.worker_init_fn
    )

    # test
    # zzh HERE
    val_stats = test(model, dataloader_test, device,
                     logger, img_out_dir, args.eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'E-NeRV training and evaluation script', parents=[get_args_parse()])
    args = parser.parse_args()

    assert args.cfg_path is not None, 'Need a specific cfg path!'
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
