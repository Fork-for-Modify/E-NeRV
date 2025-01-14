import cv2
import torch
import utils.misc as utils
import torch.nn.functional as F
from datetime import datetime


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    cfg,
    args,
    datasize,
    start_time,
    logger,
    writer=None,
):
    model.train()
    epoch_start_time = datetime.now()
    loss_type = cfg["loss"]

    psnr_list = []
    msssim_list = []

    for i, data in enumerate(dataloader):
        data = utils.to_cuda(data, device)
        # forward pass
        # output is a list for the case that has multiscale
        output_list = model(data)
        additional_loss_item = {}
        if isinstance(output_list, dict):
            for k, v in output_list.items():
                if "loss" in k:
                    additional_loss_item[k] = v
            output_list = output_list["output_list"]
        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list
        ]
        loss_list = utils.loss_compute(output_list, target_list, loss_type)
        losses = sum(loss_list)
        if len(additional_loss_item.values()) > 0:
            losses = losses + sum(additional_loss_item.values())

        lr = utils.adjust_lr(optimizer, epoch, cfg["epoch"], i, datasize, cfg)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))
        msssim_list.append(utils.msssim_fn(output_list, target_list))

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:
            train_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            train_psnr = torch.mean(train_psnr, dim=0)  # (num_stage)
            # (batchsize, num_stage)
            train_msssim = torch.cat(msssim_list, dim=0)
            train_msssim = torch.mean(
                train_msssim.float(), dim=0)  # (num_stage)
            if not hasattr(args, "rank"):
                print_str = "Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
                    epoch + 1,
                    cfg["epoch"],
                    i + 1,
                    len(dataloader),
                    lr,
                    utils.RoundTensor(train_psnr, 2, False),
                    utils.RoundTensor(train_msssim, 4, False),
                )
                for k, v in additional_loss_item.items():
                    print_str += f", {k}: {v.item():.6g}"
                logger.info(print_str)

            elif args.rank in [0, None]:
                print_str = "Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}".format(
                    args.rank,
                    epoch + 1,
                    cfg["epoch"],
                    i + 1,
                    len(dataloader),
                    lr,
                    utils.RoundTensor(train_psnr, 2, False),
                    utils.RoundTensor(train_msssim, 4, False),
                )
                logger.info(print_str)

    train_stats = {
        "train_psnr": train_psnr,
        "train_msssim": train_msssim,
    }
    if hasattr(args, "distributed") and args.distributed:
        train_stats = utils.reduce_dict(train_stats)

    # ADD train performance TO TENSORBOARD
    if not hasattr(args, "rank"):
        h, w = output_list[-1].shape[-2:]
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(
            ), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
        for k, v in additional_loss_item.items():
            writer.add_scalar(f"Train/{k}", v.item(), epoch + 1)
        for (k, m) in model.named_modules():
            if isinstance(m, torch.nn.Module) and hasattr(m, "Lip_c"):
                writer.add_scalar(f"Stat/{k}_c", m.Lip_c[0].item(), epoch + 1)
                writer.add_scalar(f"Stat/{k}_w", m.abssum_max, epoch + 1)

        writer.add_image('train/image_in', output_list[0][0].cpu(), epoch+1)
        writer.add_image('train/image_gt', target_list[0][0].cpu(), epoch+1)

    elif args.rank in [0, None] and writer is not None:
        h, w = output_list[-1].shape[-2:]
        writer.add_scalar(
            f"Train/PSNR_{h}X{w}", train_stats["train_psnr"][-1].item(), epoch + 1
        )
        writer.add_scalar(
            f"Train/MSSSIM_{h}X{w}", train_stats["train_msssim"][-1].item(
            ), epoch + 1
        )
        writer.add_scalar("Train/lr", lr, epoch + 1)
        writer.add_image('train/image_in', output_list[0][0].cpu(), epoch+1)
        writer.add_image('train/image_gt', target_list[0][0].cpu(), epoch+1)

    epoch_end_time = datetime.now()
    logger.info(
        "-> time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format(
            (epoch_end_time - epoch_start_time).total_seconds(),
            (epoch_end_time - start_time).total_seconds() / (epoch + 1),
        )
    )

    return train_stats


@torch.no_grad()
def evaluate(model, dataloader, device, cfg, args, epoch, logger, save_img=False, img_out_dir=None):
    # todo: save image
    val_start_time = datetime.now()
    model.eval()

    psnr_list = []
    msssim_list = []

    for i, data in enumerate(dataloader):
        data = utils.to_cuda(data, device)
        # forward pass
        # output is a list for the case that has multiscale
        output_list = model(data)
        if isinstance(output_list, dict):
            output_list = output_list["output_list"]  # ignore the loss in eval
        torch.cuda.synchronize()
        target_list = [
            F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list
        ]

        # compute psnr and msssim
        psnr_list.append(utils.psnr_fn(output_list, target_list))
        msssim_list.append(utils.msssim_fn(output_list, target_list))

        if i % cfg["print_freq"] == 0 or i == len(dataloader) - 1:
            val_psnr = torch.cat(psnr_list, dim=0)  # (batchsize, num_stage)
            val_psnr = torch.mean(val_psnr, dim=0)  # (num_stage)
            # (batchsize, num_stage)
            val_msssim = torch.cat(msssim_list, dim=0)
            val_msssim = torch.mean(val_msssim.float(), dim=0)  # (num_stage)
            if not hasattr(args, "rank"):
                print_str = "Eval, Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                logger.info(print_str)

            elif args.rank in [0, None]:
                print_str = "Rank:{}, Eval, Step [{}/{}], PSNR: {}, MSSSIM: {}".format(
                    args.rank,
                    i + 1,
                    len(dataloader),
                    utils.RoundTensor(val_psnr, 2, False),
                    utils.RoundTensor(val_msssim, 4, False),
                )
                logger.info(print_str)

    # save image to folder
    if save_img and img_out_dir:
        img_i = 255*output_list[0][0].cpu().numpy()[::-1].transpose(1, 2, 0)
        cv2.imwrite(f'{img_out_dir}/img_recon_e{epoch:03d}.png',
                    img_i, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    val_stats = {
        "val_psnr": val_psnr,
        "val_msssim": val_msssim,
    }
    if hasattr(args, "distributed") and args.distributed:
        val_stats = utils.reduce_dict(val_stats)
    val_end_time = datetime.now()
    logger.info(
        "-> total time on evaluate: \t{:.2f}".format(
            (val_end_time - val_start_time).total_seconds()
        )
    )

    return val_stats


@torch.no_grad()
def test(model, dataloader, device, logger, img_out_dir, eval=False):
    # todo: save image
    val_start_time = datetime.now()
    model.eval()

    psnr_list = []
    msssim_list = []
    data_num = len(dataloader)

    for i, data in enumerate(dataloader):
        data = utils.to_cuda(data, device)
        # forward pass
        # output is a list for the case that has multiscale
        assert data['img_id'].shape[0] == 1, 'batch size should equal to 1'
        output_list = model(data)
        if isinstance(output_list, dict):
            output_list = output_list["output_list"]  # ignore the loss in eval
        torch.cuda.synchronize()

        # save image to folder
        img_i = 255*output_list[0][0].cpu().numpy()[::-1].transpose(1, 2, 0)
        cv2.imwrite(f'{img_out_dir}/img_recon{i:03d}.png',
                    img_i, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # eval
        if eval:
            target_list = [
                F.adaptive_avg_pool2d(data["img_gt"], x.shape[-2:]) for x in output_list
            ]

            # compute psnr and msssim
            psnr_i = utils.psnr_fn(output_list, target_list)[0, 0]
            msssim_i = utils.msssim_fn(output_list, target_list)[0, 0]

            logger.info(
                "Frame #{}/{} PSNR: {:.2f}, MSSSIM: {:.4f}".format(i+1, data_num, psnr_i, msssim_i))
            psnr_list.append(psnr_i)
            msssim_list.append(msssim_i)
        else:
            print(f'''--> Finish: {">"*int(i*100/data_num)}{"|":>{100-int(i*100/data_num)}s} {i+1:03d}/{data_num:03d} \r''', end="")

    # mean performance
    # (batchsize, num_stage)
    mean_psnr = torch.mean(torch.tensor(psnr_list))
    mean_psnr = utils.RoundTensor(mean_psnr, 2, False)  # (num_stage)
    # (batchsize, num_stage)
    mean_msssim = torch.mean(torch.tensor(msssim_list))
    mean_msssim = utils.RoundTensor(mean_msssim, 4, False)  # (num_stage)
    logger.info("\n\nAverage PSNR: {}, MSSSIM: {}".format(
        mean_psnr, mean_msssim))

    val_stats = {
        "psnr_list": psnr_list,
        "msssim_list": msssim_list,
        "mean_psnr": mean_psnr,
        "mean_msssim": mean_msssim
    }

    val_end_time = datetime.now()
    logger.info(
        "-> total time on test: \t{:.2f}".format(
            (val_end_time - val_start_time).total_seconds()
        )
    )

    return val_stats
