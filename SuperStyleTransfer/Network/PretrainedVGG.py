from .NetworkBase import NetworkBase


class PretrainedVGG(NetworkBase):
    """
    This should be a pretrained VGG16
    should use torchvision.models.vgg16(pretrained=True)
    """
    def __init__(self):
        super(PretrainedVGG, self).__init__()
        raise NotImplementedError
