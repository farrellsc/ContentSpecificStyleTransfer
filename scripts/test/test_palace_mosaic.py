import torch
from SuperStyleTransfer.Utils.Utils import *
from torchvision import transforms
import SuperStyleTransfer.Utils.Utils as utils
from SuperStyleTransfer.Model.JohnsonNet import JohnsonNet
from SuperStyleTransfer.Utils.DotDict import DotDict
import pickle
import os


def classify(args):
    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).cuda()

    with torch.no_grad():
        style_model = JohnsonNet(args)
        style_model.load_model(args.model)
        style_model.set_input(content_image)
        output = style_model.test()
    utils.save_image(args.output_image, output[0].cpu())


if __name__ == '__main__':
    groups = ("alldata_test",)
    models = (
#        "Mon_Dec__3_00:20:41_2018_mosaic_1e5_1e10_1_palace1000_3",
#        "Mon_Dec__3_00:33:44_2018_mosaic_1e5_1e10_1_palace1000_6",
        "Mon_Dec__3_13:50:17_2018_candy_1e5_1e10_1_alldata",
    )

    for group in groups:
        for model in models:
            outputdir = "../../output/JohnsonNet/" + group + "/" + model + "/"
            contentdir = "../../data/trainingData/" + group + "/" + group + "/"
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            for image in os.listdir(contentdir)[:500]:
                args = {
                    "blocknum": 6,
                    "content_image": contentdir + image,
                    "content_scale": None,
                    "model": "../../models/test/" + model + ".model",
                    "output_image": outputdir + image
                }
                with torch.cuda.device(0):
                    classify(DotDict(args))
