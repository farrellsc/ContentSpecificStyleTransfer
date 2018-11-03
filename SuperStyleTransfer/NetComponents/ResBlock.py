import torch.nn as nn
import torch
from .ConvLayer import ConvLayer


class ResBlock(nn.Module):
    """
    component in JohnsonNet
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        :param x: batch input data
        :return: batch output data
        """
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out
