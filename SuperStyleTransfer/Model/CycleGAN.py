from .BaseModel import BaseModel
import itertools
from overrides import overrides
import torch
from SuperStyleTransfer.NetComponents.GanLoss import GanLoss
from SuperStyleTransfer.Data.DataLoader import DataLoader


class CycleGAN(BaseModel):
    def __init__(self, args):
        super(CycleGAN, self).__init__()
        self.args = args
        self.G = self.construct_generator()
        self.F = self.construct_generator()
        self.Dx = self.construct_discriminator()
        self.Dy = self.construct_discriminator()
        self.GanLoss = GanLoss()
        self.CycleLoss = torch.nn.L1Loss()

        raise NotImplementedError

    def construct_generator(self):
        raise NotImplementedError

    def construct_discriminator(self):
        raise NotImplementedError

    @overrides
    def initialize_model(self):
        self.optimizer_generator = torch.optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()),
                                            lr=self.args.lr, betas=(self.args.beta1, 0.999))
        self.optimizer_discriminator = torch.optim.Adam(itertools.chain(self.Dx.parameters(), self.Dy.parameters()),
                                            lr=self.args.lr, betas=(self.args.beta1, 0.999))
        raise NotImplementedError

    @overrides
    def forward(self):
        raise NotImplementedError

    @overrides
    def backward(self):
        self.backward_discriminator()
        self.optimizer_generator.step()
        self.backward_generator()
        self.optimizer_discriminator.step()

    def backward_discriminator(self):
        raise NotImplementedError

    def backward_generator(self):
        raise NotImplementedError

    @overrides
    def set_input(self, x):
        raise NotImplementedError

    @overrides
    def optimize_parameters(self):
        raise NotImplementedError

    @overrides
    def test(self):
        raise NotImplementedError
