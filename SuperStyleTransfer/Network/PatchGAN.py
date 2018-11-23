from .NetworkBase import NetworkBase
import torch.nn as nn


class PatchGAN(NetworkBase):
    def __init__(self, in_channel_num, channel_base_num=64, layer_num=3):
        super(PatchGAN, self).__init__()
        norm_layer = nn.InstanceNorm2d

        kernel_width = 4
        padding_width = 1
        sequence = [
            nn.Conv2d(in_channel_num, channel_base_num, kernel_size=kernel_width, stride=2, padding=padding_width),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, layer_num):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(channel_base_num * nf_mult_prev, channel_base_num * nf_mult,
                          kernel_size=kernel_width, stride=2, padding=padding_width, bias=True),
                norm_layer(channel_base_num * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** layer_num, 8)
        sequence += [
            nn.Conv2d(channel_base_num * nf_mult_prev, channel_base_num * nf_mult,
                      kernel_size=kernel_width, stride=1, padding=padding_width, bias=True),
            norm_layer(channel_base_num * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(channel_base_num * nf_mult, 1, kernel_size=kernel_width, stride=1, padding=padding_width)]

        sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
