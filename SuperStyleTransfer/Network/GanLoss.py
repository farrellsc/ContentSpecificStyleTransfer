from torch import nn


class GanLoss(nn.Module):
    def __init__(self):
        super(GanLoss, self).__init__()
        raise NotImplementedError
