import torch.nn as nn
from SuperStyleTransfer.NetComponents.ResBlock import ResBlock


class ResnetGenerator(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, channel_base_num=64, n_blocks=6, padding_type='reflect'):
        """
        :param in_channel_num: input number of channels
        :param out_channel_num: output number of channels
        :param channel_base_num:
        :param n_blocks:
        :param padding_type:
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.in_channel_num = in_channel_num
        self.out_channel_num = out_channel_num
        self.channel_base_num = channel_base_num

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channel_num, channel_base_num, kernel_size=7, padding=0),
                 nn.ReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(channel_base_num * mult, channel_base_num * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.ReLU()]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            # We set norm_layer and use_bias as two default values. This is different from the ResBlock in the
            # original implementation
            # Note that use_bias can only be True if
            model += [ResBlock(channel_base_num * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(channel_base_num * mult, int(channel_base_num * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.ReLU()]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(channel_base_num, out_channel_num, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
