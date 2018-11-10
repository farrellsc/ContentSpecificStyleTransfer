import torch


class BaseModel:
    def __init__(self):
        super(BaseModel, self).__init__()
        self.total_loss = None
        self.content_loss = None
        self.style_loss = None

    def initialize_model(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def set_input(self, *data):
        raise NotImplementedError

    def optimize_parameters(self):
        """call forward and backward to implement a step"""
        raise NotImplementedError

    def get_current_loss(self):
        return self.content_loss, self.style_loss
    
    def test(self):
        raise NotImplementedError

