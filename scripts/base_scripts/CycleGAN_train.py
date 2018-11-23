"""
TODO: To implement CycleGAN
"""
import os
import torch
import numpy as np
import time
from SuperStyleTransfer.Utils.DotDict import DotDict
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from SuperStyleTransfer.Data.UnalignedDataSet import UnalignedDataset
from SuperStyleTransfer.Model.CycleGAN import CycleGAN


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_dataset = datasets.ImageFolder(args.dataset, transform)
    style_dataset = datasets.ImageFolder(args.styleset, transform)
    synth_dataset = UnalignedDataset(content_dataset, style_dataset)
    train_loader = DataLoader(synth_dataset, batch_size=args.batch_size)
    print("data loaded, data size:", len(synth_dataset))

    CycleGanModel = CycleGAN(args)
    CycleGanModel.initialize_model()

    for e in range(args.epochs):
        agg_loss_all = 0.
        agg_loss_cycle_GF = 0.
        agg_loss_cycle_FG = 0.
        agg_loss_Dx = 0.
        agg_loss_Dy = 0.
        agg_loss_G = 0.
        agg_loss_F = 0.
        count = 0
        for batch_id, (A, B) in enumerate(train_loader):
            CycleGanModel.set_input(A, B)
            count += CycleGanModel.args.n_batch
            CycleGanModel.optimize_parameters()

            losses = CycleGanModel.get_current_loss()
            agg_loss_all += losses[0].item()
            agg_loss_cycle_GF += losses[1].item()
            agg_loss_cycle_FG += losses[2].item()
            agg_loss_Dx += losses[3].item()
            agg_loss_Dy += losses[4].item()
            agg_loss_G += losses[5].item()
            agg_loss_F += losses[6].item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tCycle: {:.6f}\tDiscriminator: {:.6f}\tGenerator: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, max(len(synth_dataset)), (agg_loss_cycle_GF+agg_loss_cycle_FG) / (batch_id + 1),
                    (agg_loss_Dx+agg_loss_Dy) / (batch_id + 1), (agg_loss_G+agg_loss_F) / (batch_id+1),
                    agg_loss_all / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                CycleGanModel.save_model(ckpt_model_path)

    # save model
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    CycleGanModel.save_model(save_model_path)

    print("\nDone, trained model saved at", save_model_path)


if __name__ == '__main__':
    args = {
        "seed": 42,
        "image_size": 256,
        "batch_size": 16,
        "epochs": 1,
        "log_interval": 50,
        "checkpoint_model_dir": None,
        "checkpoint_interval": 200,
        "dataset": "../../data/images/content-images/",
        "styleset": "../../data/images/style-images2/",
        "save_model_dir": "../../models/CycleGan/",

        "in_channel_num": 3,
        "out_channel_num": 3,
        "channel_base_num": 64,
        "netG_type": 'resnet_6blocks',                 # 'resnet_9blocks' / 'resnet_6blocks' / 'unet_128' / 'unet_256'
        "netD_type": 'pixel',                 # 'patchGan' / 'pixel'
        "lr": 0.0002,
        "beta1": 0.5,
        "netD_layer_num": 3,
        "pool_size": 50,
        "lambda_A": 10,
        "lambda_B": 10,
        "direction": 'AtoB'             # 'AtoB' / 'BtoA'
    }
    train(DotDict(args))
