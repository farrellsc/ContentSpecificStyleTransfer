import functools
import torch
from torch import nn


class UnetGenerator(nn.Module):
    def __init__(self, in_channel_num, out_channel_num, num_downs, channel_base_num=64, norm_layer=nn.BatchNorm2d):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(channel_base_num * 8, channel_base_num * 8, in_channel_num=None,
                                             submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(channel_base_num * 8, channel_base_num * 8, in_channel_num=None,
                                                 submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(channel_base_num * 4, channel_base_num * 8, in_channel_num=None,
                                             submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(channel_base_num * 2, channel_base_num * 4, in_channel_num=None,
                                             submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(channel_base_num, channel_base_num * 2, in_channel_num=None,
                                             submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(out_channel_num, channel_base_num, in_channel_num=in_channel_num,
                                             submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, x):
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, in_channel_num=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if in_channel_num is None:
            in_channel_num = outer_nc
        downconv = nn.Conv2d(in_channel_num, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up + [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
