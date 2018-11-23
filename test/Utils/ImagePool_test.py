from SuperStyleTransfer.Utils.BaseTestCase import BaseTestCase
from SuperStyleTransfer.Utils.DotDict import DotDict
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from SuperStyleTransfer.Data.UnalignedDataSet import UnalignedDataset


class TestImagePool(BaseTestCase):
    def setUp(self):
        self.args = DotDict({
            "image_size": 256,
            "batch_size": 1,
            "dataset": "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/data/images/content-images",
            "styleset": "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/data/images/style-images/"
        })

    def test_image_pool(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_dataset = datasets.ImageFolder(self.args.dataset, transform)
        style_dataset = datasets.ImageFolder(self.args.styleset, transform)
        synth_dataset = UnalignedDataset(content_dataset, style_dataset)
        train_loader = DataLoader(synth_dataset, batch_size=self.args.batch_size)
