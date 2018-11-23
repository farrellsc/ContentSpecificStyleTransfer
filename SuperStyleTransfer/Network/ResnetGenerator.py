import torch.nn as nn
from SuperStyleTransfer.NetComponents.ResBlock import ResBlock


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """

        :param input_nc: input number of channels
        :param output_nc: output number of channels
        :param ngf:
        :param use_dropout:
        :param n_blocks:
        :param padding_type:
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            # We set norm_layer and use_bias as two default values. This is different from the ResBlock in the
            # original implementation
            # Note that use_bias can only be True if
            model += [ResBlock(ngf * mult, padding_type=padding_type, use_dropout=use_dropout, norm_layer=nn.BatchNorm2d, use_bias=False)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
