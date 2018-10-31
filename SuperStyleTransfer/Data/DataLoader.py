import torch


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self):
        super(dataLoader, self).__init__()
        raise NotImplementedError
