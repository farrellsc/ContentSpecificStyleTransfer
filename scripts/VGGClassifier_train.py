import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from SuperStyleTransfer.Model.VGGClassifier import VGGClassifier
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
    vgg = VGGClassifier(args)

    for e in range(args.epochs):
        for batch_id, (x, y) in enumerate(train_loader):
            vgg.set_input(x, y)
            vgg.optimize_parameters()

    # save model
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "vgg_classifier.model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    vgg.save_model(save_model_path)

    print("\nDone, trained model saved at", save_model_path)


if __name__ == '__main__':
    args = {
        "num_classes": 3,
        "vgg_relu_level": 1,
        "seed": 42,
        "image_size": 256,
        "batch_size": 4,
        "lr": 1e-3,
        "epochs": 2,
        "dataset": "../data/images/content-images/",
        "save_model_dir": "../models/JohnsonNet/"
    }
    train(DotDict(args))
