import torch.nn as nn
import torch


class UpSamplingLayer(nn.Module):
    """
    component in JohnsonNet
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpSamplingLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        :param x: batch input data
        :return: batch output data
        """
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
