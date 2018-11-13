import torch
import pickle
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from SuperStyleTransfer.Model.JohnsonNet import JohnsonNet
from SuperStyleTransfer.Utils.DotDict import DotDict


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    print("data loaded, data size:", len(train_dataset))

    JohnsonModel = JohnsonNet(args)
    JohnsonModel.initialize_model()

    for e in range(args.epochs):
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            JohnsonModel.set_input(x.cuda())
            count += JohnsonModel.args.n_batch
            JohnsonModel.optimize_parameters()

            current_content_loss, current_style_loss = JohnsonModel.get_current_loss()
            agg_content_loss += current_content_loss.item()
            agg_style_loss += current_style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset), agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1), (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                JohnsonModel.save_model(ckpt_model_path)

    # save model
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
        args.content_weight) + "_" + str(args.style_weight) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    JohnsonModel.save_model(save_model_path)

    print("\nDone, trained model saved at", save_model_path)


if __name__ == '__main__':
    args = {
        "seed": 42,
        "image_size": 256,
        "batch_size": 16,
        "lr": 1e-3,
        "style_image": "../../data/images/style-images/mosaic.jpg",
        "style_size": None,
        "epochs": 10,
        "content_weight": 1e5,
        "style_weight": 1e10,
        "vgg_relu_level": 1,   # 0/1/2/3
        "log_interval": 50,
        "checkpoint_model_dir": None,
        "checkpoint_interval": 200,
        "dataset": "../../data/trainingData/mileStoneData/",
        "save_model_dir": "../../models/JohnsonNet/"
    }
    with torch.cuda.device(0):
        train(DotDict(args))
