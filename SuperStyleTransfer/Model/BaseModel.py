import torch


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError