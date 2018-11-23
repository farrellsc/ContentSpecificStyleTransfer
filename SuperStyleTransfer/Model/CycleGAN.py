from SuperStyleTransfer.Network.UnetGenerator import UnetGenerator
from .BaseModel import BaseModel
import itertools
from overrides import overrides
import torch
from SuperStyleTransfer.NetComponents.GanLoss import GanLoss
from SuperStyleTransfer.Utils.ImagePool import ImagePool
from SuperStyleTransfer.Network.ResnetGenerator import ResnetGenerator
from SuperStyleTransfer.Network.PixelDiscriminator import PixelDiscriminator
from SuperStyleTransfer.Network.PatchGAN import PatchGAN


class CycleGAN(BaseModel):
    def __init__(self, args):
        super(CycleGAN, self).__init__()
        self.args = args
        self.G = self.construct_generator(self.args.in_channel_num, self.args.out_channel_num,
                                          self.args.channel_base_num, self.args.netG_type)
        self.F = self.construct_generator(self.args.in_channel_num, self.args.out_channel_num,
                                          self.args.channel_base_num, self.args.netG_type)
        self.Dx = None
        self.Dy = None

        self.optimizers = []
        self.optimizer_generator = None
        self.optimizer_discriminator = None

        self.AdvLoss = GanLoss()
        self.CycleLoss = torch.nn.L1Loss()
        self.loss_all = None
        self.loss_cycle_GF = None
        self.loss_cycle_FG = None
        self.loss_Dx = None
        self.loss_Dy = None
        self.loss_G = None
        self.loss_F = None

        self.real_A = None
        self.real_B = None
        self.fake_A = None
        self.fake_A_pool = None
        self.fake_B = None
        self.fake_B_pool = None
        self.rec_A = None
        self.rec_B = None

    def construct_generator(self, in_channel_num, out_channel_num, channel_base_num, net_type):
        if net_type == 'resnet_9blocks':
            net = ResnetGenerator(in_channel_num, out_channel_num, channel_base_num, n_blocks=9).cuda()
        elif net_type == 'resnet_6blocks':
             net = ResnetGenerator(in_channel_num, out_channel_num, channel_base_num, n_blocks=6).cuda()
        elif net_type == 'unet_128':
             net = UnetGenerator(in_channel_num, out_channel_num, 7, channel_base_num).cuda()
        elif net_type == 'unet_256':
             net = UnetGenerator(in_channel_num, out_channel_num, 8, channel_base_num).cuda()
        else:
             raise NotImplementedError('Generator model name [%s] is not recognized' % net_type)
        return net

    def construct_discriminator(self, in_channel_num, channel_base_num, net_type, layer_num=3):
        net = None
        if net_type == 'patchGan':
            net = PatchGAN(in_channel_num, channel_base_num=channel_base_num, layer_num=layer_num).cuda()
        elif net_type == 'pixel':
            net = PixelDiscriminator(in_channel_num, channel_base_num=channel_base_num).cuda()
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
        return net

    @overrides
    def initialize_model(self):
        self.Dx = self.construct_discriminator(self.args.in_channel_num + self.args.out_channel_num, self.args.channel_base_num,
                                               self.args.netD_type, self.args.netD_layer_num).cuda()
        self.Dy = self.construct_discriminator(self.args.in_channel_num + self.args.out_channel_num, self.args.channel_base_num,
                                               self.args.netD_type, self.args.netD_layer_num).cuda()
        self.optimizer_generator = torch.optim.Adam(itertools.chain(self.G.parameters(), self.F.parameters()),
                                                    lr=self.args.lr, betas=(self.args.beta1, 0.999))
        self.optimizer_discriminator = torch.optim.Adam(itertools.chain(self.Dx.parameters(), self.Dy.parameters()),
                                                        lr=self.args.lr, betas=(self.args.beta1, 0.999))
        self.optimizers.append(self.optimizer_generator)
        self.optimizers.append(self.optimizer_discriminator)

        # load/define networks

        self.fake_A_pool = ImagePool(self.args.pool_size)
        self.fake_B_pool = ImagePool(self.args.pool_size)

    @overrides
    def forward(self):
        self.fake_B = self.G(self.real_A)
        self.rec_A = self.F(self.fake_B)

        self.fake_A = self.F(self.real_B)
        self.rec_B = self.G(self.fake_A)

    @overrides
    def backward(self):
        raise NotImplementedError

    def backward_discriminator(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_Dx = self.backward_discriminator_base(self.Dx, self.real_B, fake_B)

        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_Dy = self.backward_discriminator_base(self.Dy, self.real_A, fake_A)

    def backward_discriminator_base(self, netD, real, fake):
        # Real
        loss_D_real = self.AdvLoss(netD(real), True)
        # Fake
        loss_D_fake = self.AdvLoss(netD(fake.detach()), False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_generator(self):
        # GAN loss Dx(G(A))
        self.loss_G = self.AdvLoss(self.Dx(self.fake_B), True)
        # GAN loss Dy(F(B))
        self.loss_F = self.AdvLoss(self.Dy(self.fake_A), True)
        # Forward cycle loss
        self.loss_cycle_GF = self.CycleLoss(self.rec_A, self.real_A) * self.args.lambda_A
        # Backward cycle loss
        self.loss_cycle_FG = self.CycleLoss(self.rec_B, self.real_B) * self.args.lambda_B
        # combined loss
        self.loss_all = self.loss_G + self.loss_F + self.loss_cycle_GF + self.loss_cycle_FG
        self.loss_all.backward()

    @overrides
    def set_input(self, A, B):
        AtoB = self.args.direction == 'AtoB'
        self.real_A = A if AtoB else B
        self.real_B = B if AtoB else A

    @overrides
    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.Dx, self.Dy], False)
        self.optimizer_generator.zero_grad()
        self.backward_generator()
        self.optimizer_generator.step()
        # D_A and D_B
        self.set_requires_grad([self.Dx, self.Dy], True)
        self.optimizer_discriminator.zero_grad()
        self.backward_discriminator()
        self.optimizer_discriminator.step()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @overrides
    def test(self):
        with torch.no_grad():
            return self.G(self.real_A)

    @overrides
    def get_current_loss(self):
        return self.loss_all, self.loss_cycle_GF, self.loss_cycle_FG, self.loss_Dx, self.loss_Dy, self.loss_G, self.loss_F

    @overrides
    def save_model(self, path):
        torch.save({
            "G": self.G.state_dict(),
            "F": self.F.state_dict(),
            "Dx": self.Dx.state_dict(),
            "Dy": self.Dy.state_dict()
        }, path)

    @overrides
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.G.load_state_dict(checkpoint['G'])
        self.F.load_state_dict(checkpoint['F'])
        self.Dx.load_state_dict(checkpoint['Dx'])
        self.Dy.load_state_dict(checkpoint['Dy'])
