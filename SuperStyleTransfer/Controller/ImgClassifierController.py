import torch, re
from .BaseController import BaseController
import SuperStyleTransfer.Utils.Utils as utils
from ..Model.VGGClassifier import VGGClassifier
from ..Model.JohnsonNet import JohnsonNet


class ImgClassifierController(BaseController):
    def __init__(self, args, loader):
        super(ImgClassifierController, self).__init__()
        self.args = args
        self.loader = loader
        self.Classifier = VGGClassifier(args)
        self.style_models = {className: JohnsonNet(args) for className in self.args.models}
        for clsname, path in self.style_models:
            self.style_models[clsname].load_model(path)

    def stylize(self):
        with torch.no_grad():
            for batch_id, (content_image, _) in enumerate(self.loader):
                self.Classifier.set_input(content_image)
                label = self.Classifier.test()
                self.style_models[label].set_input(content_image)
                output = self.style_models[label].test()
                utils.save_image(self.args.output_folder, output)
