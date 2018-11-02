import torch.nn as nn


class UpSamplingLayer(nn.Module):
    """
    component in JohnsonNet
    """
    def __init__(self):
        super(UpSamplingLayer, self).__init__()
        raise NotImplementedError

    def forward(self, *input):
        raise NotImplementedError
