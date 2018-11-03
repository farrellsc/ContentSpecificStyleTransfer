import torch


class BaseModel:
    def __init__(self):
        raise NotImplementedError

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError