from .NetworkBase import NetworkBase
import torch
from ..NetComponents.ConvLayer import ConvLayer
from ..NetComponents.ResBlock import ResBlock
from ..NetComponents.UpSamplingLayer import UpSamplingLayer


class ResNet(NetworkBase):
    def __init__(self, n_blocks):
        # Initial convolution layers
        super(ResNet, self).__init__()
        self.n_blocks = n_blocks
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res = []
        for i in range(n_blocks):
            self.res.append(ResBlock(128))
        # Upsampling Layers
        self.deconv1 = UpSamplingLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpSamplingLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        :param x: batch input data
        :return: batch output data
        """
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        for i in range(self.n_blocks):
            y = self.res[i](y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y
