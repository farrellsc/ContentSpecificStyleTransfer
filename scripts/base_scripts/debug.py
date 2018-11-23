import SuperStyleTransfer.Utils.Utils as utils
from torchvision import transforms


content_image1 = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/data/portrait/39883201_1966-06-28_2011.jpg"
content_image1 = utils.load_image(content_image1, scale=None)
content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
content_image1 = content_transform(content_image1)
print(content_image1.shape)


content_image2 = "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/data/data 2/data 2/wild/mountain/00004170.jpg"
content_image2 = utils.load_image(content_image2, scale=None)
content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
content_image2 = content_transform(content_image2)
print(content_image2.shape)
