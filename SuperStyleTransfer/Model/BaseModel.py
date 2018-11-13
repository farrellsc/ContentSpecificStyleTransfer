class BaseModel:
    def __init__(self):
        super(BaseModel, self).__init__()

    def initialize_model(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def set_input(self, *data):
        raise NotImplementedError

    def optimize_parameters(self):
        raise NotImplementedError

    def get_current_loss(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError
