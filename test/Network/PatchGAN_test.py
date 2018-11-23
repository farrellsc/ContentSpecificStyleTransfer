from SuperStyleTransfer.Utils.BaseTestCase import BaseTestCase
from SuperStyleTransfer.Utils.DotDict import DotDict
from SuperStyleTransfer.Network.PatchGAN import PatchGAN
from SuperStyleTransfer.NetComponents.GanLoss import GanLoss
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from SuperStyleTransfer.Data.UnalignedDataSet import UnalignedDataset


class TestPatchGAN(BaseTestCase):
    def setUp(self):
        self.args = DotDict({
            "image_size": 256,
            "batch_size": 1,
            "dataset": "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/data/images/content-images",
            "styleset": "/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/SuperStyleTransfer/data/images/style-images/",
            "in_channel_num": 3,
            "channel_base_num": 64,
            "layer_num": 3
        })
        self.model = PatchGAN(self.args.in_channel_num, channel_base_num=self.args.channel_base_num,
                              layer_num=self.args.layer_num)
        self.AdvLoss = GanLoss()

        transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_dataset = datasets.ImageFolder(self.args.dataset, transform)
        style_dataset = datasets.ImageFolder(self.args.styleset, transform)
        synth_dataset = UnalignedDataset(content_dataset, style_dataset)
        self.train_loader = DataLoader(synth_dataset, batch_size=self.args.batch_size)

    def test_patchGan(self):
        for batch_id, (A, B) in enumerate(self.train_loader):
            loss_G = self.AdvLoss(self.model(B), True)
            print(batch_id, loss_G)
