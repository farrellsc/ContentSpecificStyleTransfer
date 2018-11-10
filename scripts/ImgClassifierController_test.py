from SuperStyleTransfer.Utils.Utils import *
from torchvision import transforms
from SuperStyleTransfer.Model.JohnsonNet import JohnsonNet
import SuperStyleTransfer.Utils.Utils as utils
from SuperStyleTransfer.Controller.ImgClassifierController import ImgClassifierController
from SuperStyleTransfer.Utils.DotDict import DotDict
from torchvision import datasets
from torch.utils.data import DataLoader
import re


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    test_dataset = datasets.ImageFolder(args.dataset, transform)
    test_loader = DataLoader(test_dataset, batch_size=1)
    controller = ImgClassifierController(args, test_loader)
    controller.stylize()


if __name__ == '__main__':
    args = {
        "content_folder": "/home/zz2590/SuperStyleTransfer/data/images/content-images/TestData/",
        "content_scale": None,
        "classifier": "../models/Classifier/hehe.pth",
        "models": {
            0: "../models/JohnsonNet/candy.pth",    # 0 for landscape
            1: "../models/JohnsonNet/mosaic.pth"    # 1 for portrait
        },
        "output_folder": "/home/zz2590/SuperStyleTransfer/output/JohnsonNet/Batch/"
    }
    main(DotDict(args))
