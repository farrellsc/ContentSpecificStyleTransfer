import torch

from SuperStyleTransfer.Model import CycleGAN
from SuperStyleTransfer.Network.ResnetGenerator import ResnetGenerator
from SuperStyleTransfer.Network.UnetGenerator import UnetGenerator

print(torch.cuda.is_available())
net = ResnetGenerator(3, 3)
net = UnetGenerator(3, 3, 7)

net = UnetGenerator(3, 3, 8, 64)
