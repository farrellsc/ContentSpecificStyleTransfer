from .BaseModel import BaseModel
import itertools
import torch
from ..Network.GanLoss import GanLoss
from SuperStyleTransfer.Data.DataLoader import DataLoader


class CycleGAN(BaseModel):
    def __init__(self, dataloader: DataLoader, **kwargs):
        super(CycleGAN, self).__init__()
        self.G = self.construct_generator()
        self.F = self.construct_generator()
        self.Dx = self.construct_discriminator()
        self.Dy = self.construct_discriminator()
        self.GanLoss = GanLoss()
        self.CycleLoss = torch.nn.L1Loss()

        self.optimizer_generator = torch.optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_discriminator = torch.optim.Adam(itertools.chain(self.Dx.parameters(), self.Dy.parameters()),
                                            lr=opt.lr, betas=(opt.beta1, 0.999))
        raise NotImplementedError

    def construct_generator(self):
        raise NotImplementedError

    def construct_discriminator(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        self.backward_discriminator()
        self.optimizer_generator.step()
        self.backward_generator()
        self.optimizer_discriminator.step()

    def backward_discriminator(self):
        raise NotImplementedError

    def backward_generator(self):
        raise NotImplementedError
