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
    content_image = content_image.unsqueeze(0).cuda()

    with torch.no_grad():
        style_model = JohnsonNet()
        style_model.load_model(args.model)
        style_model.set_input(content_image)
        output = style_model.test()
    utils.save_image(args.output_image, output[0].cpu())


if __name__ == '__main__':
    groups = ("city", "mountain")
    models = ("Tue_Nov_13_07:30:48_2018_udnie_1e5_1e10_1_city.model",
              "Tue_Nov_13_08:00:32_2018_udnie_1e5_1e10_1_mountain.model",
              "Tue_Nov_13_08:30:13_2018_udnie_1e5_1e10_2_city.model",
              "Tue_Nov_13_08:59:53_2018_udnie_1e5_1e10_2_mountain.model",
              "Tue_Nov_13_09:29:32_2018_udnie_1e5_1e10_3_city.model",
              "Tue_Nov_13_09:59:11_2018_udnie_1e5_1e10_3_mountain.model",
              "Tue_Nov_13_10:29:01_2018_rain-princess-cropped_1e5_1e10_0_city.model",
              "Tue_Nov_13_10:58:50_2018_rain-princess-cropped_1e5_1e10_0_mountain.model",
              "Tue_Nov_13_11:28:35_2018_rain-princess-cropped_1e5_1e10_1_city.model",
              "Tue_Nov_13_11:58:18_2018_rain-princess-cropped_1e5_1e10_1_mountain.model",
              "Tue_Nov_13_12:28:00_2018_rain-princess-cropped_1e5_1e10_2_city.model",
              "Tue_Nov_13_12:57:40_2018_rain-princess-cropped_1e5_1e10_2_mountain.model",
              "Tue_Nov_13_13:27:20_2018_rain-princess-cropped_1e5_1e10_3_city.model",
              "Tue_Nov_13_13:56:59_2018_rain-princess-cropped_1e5_1e10_3_mountain.model")

    for group in groups:
        for model in models:
            for i in range(1, 501):
                args = {
                    "content_image": "../../data/trainingData/" + group + "/" + group + "/%08d.jpg" % i,
                    "content_scale": None,
                    "model": "../../models/JohnsonNet/" + model,
                    "output_image": "../../output/JohnsonNet/" + group + "/%08d.jpg" % i
                }
                with torch.cuda.device(0):
                    classify(DotDict(args))
