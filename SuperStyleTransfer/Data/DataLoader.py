import torch


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self):
        super(DataLoader, self).__init__()
        raise NotImplementedError
