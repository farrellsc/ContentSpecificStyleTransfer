from .BaseModel import BaseModel
from ..Network.ResNet import ResNet
from ..Network.PretrainedVGG import PretrainedVGG
from ..Utils import Utils as utils
from overrides import overrides
from torch.optim import Adam
from torchvision import transforms
import torch


class JohnsonNet(BaseModel):
    def __init__(self, args):
        # Initial convolution layers
        super(JohnsonNet, self).__init__()
        self.args = args

        self.TransformerNet = ResNet()
        self.optimizer_T = Adam(self.TransformerNet.parameters(), args.lr)
        self.lossFunc = torch.nn.MSELoss()

        self.LossNet = PretrainedVGG(requires_grad=False)
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        style = utils.load_image(args.style_image, size=args.style_size)
        style = style_transform(style)
        self.style = style.repeat(args.batch_size, 1, 1, 1)
        features_style = self.LossNet(utils.normalize_batch(style))
        self.gram_style = [utils.calc_gram_matrix(y) for y in features_style]

        self.x = None
        self.y = None
        self.features_y = None
        self.features_x = None

    @overrides
    def forward(self, x):
        y = self.TransformerNet(x)
        self.y = utils.normalize_batch(y)
        self.x = utils.normalize_batch(x)
        self.features_y = self.LossNet(y)
        self.features_x = self.LossNet(x)

    @overrides
    def backward(self):
        self.content_loss = self.args.content_weight * self.lossFunc(self.features_y.relu2_2, self.features_x.relu2_2)

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

    @overrides
    def optimize_parameters(self):
        self.forward()
        self.optimizer_T.zero_grad()
        self.backward()
        self.optimizer_T.step()

