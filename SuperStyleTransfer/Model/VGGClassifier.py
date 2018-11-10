from .BaseModel import BaseModel
import torch
from torch.optim import Adam
from overrides import overrides
from SuperStyleTransfer.Network.CroppedVGG import CroppedVGG
from SuperStyleTransfer.Utils import Utils as utils
from SuperStyleTransfer.Utils.DotDict import DotDict


class VGGClassifier(BaseModel):
    """
    This should be a pretrained VGG16
    should use torchvision.models.vgg16(pretrained=True)

    Given a input sample, it outputs four results of different depth in network
    From shallow to deep are relu1, relu2, relu3, relu3
    """

    def __init__(self, args=DotDict({}), requires_grad=False):
        super(VGGClassifier, self).__init__()
        self.args = args
        self.ConvNet = CroppedVGG(requires_grad=False)
        self.Classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, self.args.num_classes),
        )
        self.optimizer = Adam(self.Classifier.parameters(), self.args.lr)
        self.lossFunc = torch.nn.MSELoss()
        self.x = None
        self.y = None
        self.pred = None

    @overrides
    def forward(self):
        """
        :param x: batch input data
        :return: batch output at four different depth in network
        """
        x = self.ConvNet(self.x)[self.args.vgg_relu_level]
        x = x.view(x.size(0), -1)
        self.pred = self.Classifier(x)

    @overrides
    def backward(self):
        self.total_loss = self.lossFunc(self.pred, self.y)
        self.total_loss.backward()

    @overrides
    def set_input(self, x, y=None):
        self.x = x
        self.y = y

    @overrides
    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    @overrides
    def test(self):
        with torch.no_grad():
            self.forward()
            return self.pred

    @overrides
    def save_model(self, path):
        torch.save({
            "ConvNet": self.ConvNet.state_dict(),
            "Classifier": self.Classifier.state_dict()
        }, path)

    @overrides
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.ConvNet.load_state_dict(checkpoint['ConvNet'])
        self.Classifier.load_state_dict(checkpoint['Classifier'])
