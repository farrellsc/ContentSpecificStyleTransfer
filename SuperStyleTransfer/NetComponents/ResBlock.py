import torch.nn as nn


class ResBlock(nn.Module):
    """
    component in JohnsonNet
    """
    def __init__(self):
        super(ResBlock, self).__init__()
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError
