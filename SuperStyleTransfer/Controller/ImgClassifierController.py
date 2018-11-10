import torch, re
from .BaseController import BaseController
import SuperStyleTransfer.Utils.Utils as utils

from ..Model.JohnsonNet import JohnsonNet


class ImgClassifierController(BaseController):
    def __init__(self, args, loader):
        super(ImgClassifierController, self).__init__()
        self.args = args
        self.loader = loader
        self.Classifier = torch.load(args.classifier)
        self.state_dicts = {className: torch.load(path) for className, path in args.models}
        self.style_models = {className: JohnsonNet(args) for className in self.state_dicts.keys()}
        for className in self.state_dicts.keys():
            for k in list(self.state_dicts[className].keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del self.state_dicts[className][k]
            self.style_models[className].TransformerNet.load_state_dict(self.state_dicts[className])

    def stylize(self):
        with torch.no_grad():
            for batch_id, (content_image, _) in enumerate(self.loader):
                label = self.Classifier(content_image)
                self.style_models[label].set_input(content_image)
                output = self.style_models[label].test()
                utils.save_image(self.args.output_folder, output[0])
