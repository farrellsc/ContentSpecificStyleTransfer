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
        self.n_blocks = n_blocks
        super(ResnetGenerator, self).__init__()
        self.in_channel_num = in_channel_num
        self.out_channel_num = out_channel_num
        self.channel_base_num = channel_base_num

        self.model_head = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channel_num, channel_base_num, kernel_size=7, padding=0),
                 nn.ReLU()]

        n_downsampling = 2
        self.n_downsampling = n_downsampling
        self.model_downsampling = []
        for i in range(n_downsampling):
            mult = 2**i
            self.model_downsampling.append(nn.Conv2d(channel_base_num * mult, channel_base_num * mult * 2, kernel_size=3, stride=2, padding=1))
            self.model_downsampling.append(nn.ReLU())

        mult = 2**n_downsampling
        self.model_nblocks = []
        for i in range(n_blocks):
            # We set norm_layer and use_bias as two default values. This is different from the ResBlock in the
            # original implementation
            # Note that use_bias can only be True if
            self.model_nblocks.append(ResBlock(channel_base_num * mult))

        self.model_upsampling = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            self.model_upsampling.append(nn.ConvTranspose2d(channel_base_num * mult, int(channel_base_num * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1))
            self.model_upsampling.append(nn.ReLU())

        self.model_tail = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channel_base_num, out_channel_num, kernel_size=7, padding=0),
            nn.Tanh()]

    def forward(self, x):
        y = self.model_head[0](x)
        y = self.model_head[1](y)
        y = self.model_head[2](y)
        for i in range(self.n_downsampling):
            y = self.model_downsampling[i](y)
        for i in range(self.n_blocks):
            y = self.model_nblocks[i](y)
        for i in range(self.n_downsampling):
            y = self.model_upsampling[i](y)
        y = self.model_tail[0](y)
        y = self.model_tail[1](y)
        y = self.model_tail[2](y)
        return y
