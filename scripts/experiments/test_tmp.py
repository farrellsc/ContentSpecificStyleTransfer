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
        style_model = JohnsonNet()
        style_model.load_model(args.model)
        style_model.set_input(content_image)
        output = style_model.test()
    utils.save_image(args.output_image, output[0].cpu())


if __name__ == '__main__':
    groups = ("city", "mountain")
    models = ("Mon_Nov_12_22:35:39_2018_mosaic_1e5_1e10_0_city",
              "Mon_Nov_12_23:05:28_2018_mosaic_1e5_1e10_0_mountain",
              "Mon_Nov_12_23:35:11_2018_mosaic_1e5_1e10_1_city",
              "Tue_Nov_13_00:04:54_2018_mosaic_1e5_1e10_1_mountain",
              "Tue_Nov_13_00:34:34_2018_mosaic_1e5_1e10_2_city",
              "Tue_Nov_13_01:04:14_2018_mosaic_1e5_1e10_2_mountain",
              "Tue_Nov_13_01:33:53_2018_mosaic_1e5_1e10_3_city",
              "Tue_Nov_13_02:03:31_2018_mosaic_1e5_1e10_3_mountain",
              "Tue_Nov_13_02:33:22_2018_candy_1e5_1e10_0_city",
              "Tue_Nov_13_03:03:12_2018_candy_1e5_1e10_0_mountain",
              "Tue_Nov_13_03:32:57_2018_candy_1e5_1e10_1_city",
              "Tue_Nov_13_04:02:41_2018_candy_1e5_1e10_1_mountain",
              "Tue_Nov_13_04:32:23_2018_candy_1e5_1e10_2_city",
              "Tue_Nov_13_05:02:05_2018_candy_1e5_1e10_2_mountain",
              "Tue_Nov_13_05:31:45_2018_candy_1e5_1e10_3_city",
              "Tue_Nov_13_06:01:24_2018_candy_1e5_1e10_3_mountain",
              "Tue_Nov_13_06:31:15_2018_udnie_1e5_1e10_0_city",
              "Tue_Nov_13_07:01:04_2018_udnie_1e5_1e10_0_mountain",
              "Tue_Nov_13_07:30:48_2018_udnie_1e5_1e10_1_city",
              "Tue_Nov_13_08:00:32_2018_udnie_1e5_1e10_1_mountain",
              "Tue_Nov_13_08:30:13_2018_udnie_1e5_1e10_2_city",
              "Tue_Nov_13_08:59:53_2018_udnie_1e5_1e10_2_mountain",
              "Tue_Nov_13_09:29:32_2018_udnie_1e5_1e10_3_city",
              "Tue_Nov_13_09:59:11_2018_udnie_1e5_1e10_3_mountain",
              "Tue_Nov_13_10:29:01_2018_rain-princess-cropped_1e5_1e10_0_city",
              "Tue_Nov_13_10:58:50_2018_rain-princess-cropped_1e5_1e10_0_mountain",
              "Tue_Nov_13_11:28:35_2018_rain-princess-cropped_1e5_1e10_1_city",
              "Tue_Nov_13_11:58:18_2018_rain-princess-cropped_1e5_1e10_1_mountain",
              "Tue_Nov_13_12:28:00_2018_rain-princess-cropped_1e5_1e10_2_city",
              "Tue_Nov_13_12:57:40_2018_rain-princess-cropped_1e5_1e10_2_mountain",
              "Tue_Nov_13_13:27:20_2018_rain-princess-cropped_1e5_1e10_3_city",
              "Tue_Nov_13_13:56:59_2018_rain-princess-cropped_1e5_1e10_3_mountain")

    for group in groups:
        for model in models:
            for i in range(1, 501):
                outputdir = "../../output/JohnsonNet/" + group + "/" + model + "/"
                contentdir = "../../data/trainingData/" + group + "/" + group + "/"
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                for image in os.listdir(contentdir):
                    args = {
                        "content_image": contentdir + image,
                        "content_scale": None,
                        "model": "../../models/JohnsonNet/" + model + ".model",
                        "output_image": outputdir + image
                    }
                    with torch.cuda.device(0):
                        classify(DotDict(args))
