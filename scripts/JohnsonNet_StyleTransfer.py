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
        style_model = JohnsonNet()
        state_dict = torch.load(args.model)
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        output = style_model(content_image)
    utils.save_image(args.output_image, output[0])


if __name__ == '__main__':
    args = {
        "content_image": "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/data/images/content-images/amber.jpg",
        "content_scale": None,
        "model": "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/models/johnsonNet/candy.pth",
        "output_image": "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/output/johnsonNet/test.jpg"
    }
    stylize(DotDict(args))
