from .BaseModel import BaseModel
from ..Network.ResNet import ResNet
from ..Network.CroppedVGG import CroppedVGG
from ..Utils import Utils as utils
from overrides import overrides
from torch.optim import Adam
from torchvision import transforms
from SuperStyleTransfer.Utils.DotDict import DotDict
import torch


class JohnsonNet(BaseModel):
    def __init__(self, args=DotDict({})):
        # Initial convolution layers
        super(JohnsonNet, self).__init__()
        self.args = args
        self.args.n_batch = 0

        self.TransformerNet = ResNet().cuda()

        self.x = None
        self.y = None
        self.features_y = None
        self.features_x = None

    @overrides
    def initialize_model(self):
        self.optimizer_T = Adam(self.TransformerNet.parameters(), self.args.lr)
        self.lossFunc = torch.nn.MSELoss()

        self.LossNet = CroppedVGG(requires_grad=False).cuda()
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        style = utils.load_image(self.args.style_image, size=self.args.style_size)
        style = style_transform(style).cuda()
        self.style = style.repeat(self.args.batch_size, 1, 1, 1)
        features_style = self.LossNet(utils.normalize_batch(self.style))
        self.gram_style = [utils.calc_gram_matrix(y) for y in features_style]

    @overrides
    def forward(self):
        self.y = self.TransformerNet(self.x)
        self.y = utils.normalize_batch(self.y)
        self.x = utils.normalize_batch(self.x)
        self.features_y = self.LossNet(self.y)
        self.features_x = self.LossNet(self.x)

    @overrides
    def backward(self):
        self.content_loss = self.args.content_weight * self.lossFunc(self.features_y[self.args.vgg_relu_level], self.features_x[self.args.vgg_relu_level])

        self.style_loss = 0.
        for ft_y, gm_s in zip(self.features_y, self.gram_style):
            gm_y = utils.calc_gram_matrix(ft_y)
            self.style_loss += self.lossFunc(gm_y, gm_s[:self.args.n_batch, :, :])
        self.style_loss *= self.args.style_weight

        self.total_loss = self.content_loss + self.style_loss
        self.total_loss.backward()

    @overrides
    def set_input(self, x):
        self.x = x
        self.args.n_batch = len(x)

    @overrides
    def optimize_parameters(self):
        self.forward()
        self.optimizer_T.zero_grad()
        self.backward()
        self.optimizer_T.step()

    @overrides
    def test(self):
        with torch.no_grad():
            return self.TransformerNet(self.x)

    @overrides
    def save_model(self, path):
        torch.save({
            "TransformerNet": self.TransformerNet.state_dict()
        }, path)

    @overrides
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.TransformerNet.load_state_dict(checkpoint['TransformerNet'])
