import torch
from SuperStyleTransfer.Utils.Utils import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import SuperStyleTransfer.Utils.Utils as utils
from SuperStyleTransfer.Model.JohnsonNet import JohnsonNet
from SuperStyleTransfer.Utils.DotDict import DotDict
import pickle


def classify(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    print("data loaded, data size:", len(train_dataset))

    with torch.no_grad():
        for batch_id, (content_image, _) in enumerate(train_loader):
            content_image = content_image.unsqueeze(0).cuda()
            style_model = JohnsonNet()
            style_model.load_model(args.model)
            style_model.set_input(content_image)
            output = style_model.test()
            utils.save_image(args.output_image, output[0].cpu())


if __name__ == '__main__':
    args = {
        "dataset": "../../data/trainingData/mileStoneData/",
        "content_scale": None,
        "model": "../../models/JohnsonNet/epoch_10_Sun_Nov_11_22:38:42_2018_100000.0_10000000000.0.model",
        "outputpath": "../../output/JohnsonNet/"
    }
    with torch.cuda.device(0):
        classify(DotDict(args))
