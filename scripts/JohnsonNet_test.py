import torch
from SuperStyleTransfer.Utils.Utils import *
from torchvision import transforms
import SuperStyleTransfer.Utils.Utils as utils
from SuperStyleTransfer.Model.JohnsonNet import JohnsonNet
from SuperStyleTransfer.Utils.DotDict import DotDict
import pickle


def classify(args):
    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)

    with torch.no_grad():
        style_model = JohnsonNet()
        style_model.load_model(args.model)
        style_model.set_input(content_image)
        output = style_model.test()
    utils.save_image(args.output_image, output[0])


if __name__ == '__main__':
    args = {
        "content_image": "/home/zz2590/SuperStyleTransfer/data/images/content-images/TrainingData/amber.jpg",
        "content_scale": None,
        "model": "../models/JohnsonNet/candy.pth",
        "output_image": "/home/zz2590/SuperStyleTransfer/output/JohnsonNet/test.jpg"
    }
    classify(DotDict(args))
