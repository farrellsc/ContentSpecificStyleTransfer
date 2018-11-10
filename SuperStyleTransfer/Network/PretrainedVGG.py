from .NetworkBase import NetworkBase
import torch
from torchvision import models
from typing import Dict


class PretrainedVGG(NetworkBase):
    """
    This should be a pretrained VGG16
    should use torchvision.models.vgg16(pretrained=True)

    Given a input sample, it outputs four results of different depth in network
    From shallow to deep are relu1, relu2, relu3, relu3
    """

    def __init__(self, model_path, requires_grad=False):
        super(PretrainedVGG, self).__init__()
        # vgg_pretrained_features = models.vgg16(pretrained=True).features
        vgg_pretrained_features = torch.load(model_path).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x) -> dict:
        """
        :param x: batch input data
        :return: batch output at four different depth in network
        """
        h = self.slice1(x)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        out = {
            'relu1': h_relu1,
            'relu2': h_relu2,
            'relu3': h_relu3,
            'relu4': h_relu4
        }
        return out
