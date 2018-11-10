import torch
from SuperStyleTransfer.Utils.Utils import *
from torchvision import transforms
from SuperStyleTransfer.Model.JohnsonNet import JohnsonNet
import SuperStyleTransfer.Utils.Utils as utils
from SuperStyleTransfer.Utils.DotDict import DotDict
import re


def stylize(args):
    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)

    with torch.no_grad():
        style_model = JohnsonNet(args)
        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.TransformerNet.load_state_dict(state_dict)
        style_model.set_input(content_image)
        output = style_model.test()
    utils.save_image(args.output_image, output[0])


if __name__ == '__main__':
    args = {
        "content_image": "/home/zz2590/SuperStyleTransfer/data/images/content-images/TrainingData/amber.jpg",
        "content_scale": None,
        "model": "../models/JohnsonNet/epoch_2_Sat_Nov_10_01:09:42_2018_100000.0_10000000000.0.model",
        "output_image": "/home/zz2590/SuperStyleTransfer/output/JohnsonNet/test2.jpg"
    }
    stylize(DotDict(args))
