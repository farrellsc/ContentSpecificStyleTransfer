from .NetworkBase import NetworkBase
import torch.nn as nn


class PixelDiscriminator(NetworkBase):
    def __init__(self, in_channel_num, channel_base_num=64):
        super(PixelDiscriminator, self).__init__()

        self.net = [
            nn.Conv2d(in_channel_num, channel_base_num, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channel_base_num, channel_base_num * 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channel_base_num * 2, 1, kernel_size=1, stride=1, padding=0)
        ]

        self.net.append(nn.Sigmoid())
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)
