import torch


class GanLoss(torch.nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GanLoss, self).__init__()
        self.register_buffer('real_label', torch.Tensor(target_real_label))
        self.register_buffer('fake_label', torch.Tensor(target_fake_label))
        self.loss = torch.nn.MSELoss()

    def get_target_tensor(self, x, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(x)

    def __call__(self, x, target_is_real):
        target_tensor = self.get_target_tensor(x, target_is_real)
        return self.loss(x, target_tensor)
