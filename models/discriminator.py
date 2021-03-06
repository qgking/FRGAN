import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from module.modules import BaseModule


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm3d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_normal_2d(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 70*70   patchgan2
class Kumar_discriminator(nn.Module):
    # initializers
    def __init__(self, in_channels=1, filters=64):
        super(Kumar_discriminator, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, filters, 4, 2, 1)
        self.conv2 = nn.Conv3d(filters, filters * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm3d(filters * 2)
        self.conv3 = nn.Conv3d(filters * 2, filters * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d(filters * 4)
        self.conv4 = nn.Conv3d(filters * 4, filters * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm3d(filters * 8)
        self.conv5 = nn.Conv3d(filters * 8, 1, 4, 1, 1)
        self.apply(weights_init_normal)

    # forward method
    def forward(self, input):
        x = input
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x


# 70*70   patchgan2
class Dakai_discriminator(nn.Module):
    # initializers
    def __init__(self, in_channels, filters=64):
        super(Dakai_discriminator, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, filters, 4, 2, 1)
        self.conv2 = nn.Conv3d(filters, filters * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm3d(filters * 2)
        self.conv3 = nn.Conv3d(filters * 2, filters * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d(filters * 4)
        self.conv4 = nn.Conv3d(filters * 4, filters * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm3d(filters * 8)
        self.conv5 = nn.Conv3d(filters * 8, 1, 4, 1, 1)
        self.apply(weights_init_normal)

    # forward method
    def forward(self, input):
        x = input
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))
        return x


# 46*46   patchgan3
class discriminator_v1(nn.Module):
    # initializers
    def __init__(self, in_channels=1, filters=32):
        super(discriminator_v1, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, filters, 4, 2, 1)
        self.conv2 = nn.Conv3d(filters, filters * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm3d(filters * 2)
        self.conv3 = nn.Conv3d(filters * 2, filters * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d(filters * 4)
        self.conv4 = nn.Conv3d(filters * 4, 1, 4, 1, 1)
        self.apply(weights_init_normal)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = input
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))
        return x


# 22*22  patchgan5_2D
class discriminator_v3(nn.Module):
    # initializers
    def __init__(self, in_channels=1, filters=64):
        super(discriminator_v3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters * 2, 4, 2, 1)
        self.conv2 = nn.Conv2d(filters * 2, filters * 4, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(filters * 4)
        self.conv3 = nn.Conv2d(filters * 4, 1, 4, 1, 1)
        self.apply(weights_init_normal_2d)

    # forward method
    def forward(self, input):
        x = input
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = torch.sigmoid(self.conv3(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# class SNTemporalPatchGANDiscriminator(BaseModule):
#     def __init__(
#             self, nc_in=2, nf=64, norm='SN', use_sigmoid=False, use_bias=True, conv_type='vanilla',
#             conv_by='3d'
#     ):
#         super().__init__(conv_type)
#         use_bias = use_bias
#         self.use_sigmoid = use_sigmoid
#
#         ######################
#         # Convolution layers #
#         ######################
#         # self.conv1 = self.ConvBlock(
#         #     nc_in, nf * 1, kernel_size=(5, 5, 5), stride=(2, 2, 2),
#         #     padding=1, bias=use_bias, norm=norm, conv_by=conv_by
#         # )
#         # self.conv2 = self.ConvBlock(
#         #     nf * 1, nf * 2, kernel_size=(5, 5, 5), stride=(2, 2, 2),
#         #     padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
#         # )
#         # self.conv3 = self.ConvBlock(
#         #     nf * 2, nf * 4, kernel_size=(5, 5, 5), stride=(2, 2, 2),
#         #     padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
#         # )
#         # self.conv4 = self.ConvBlock(
#         #     nf * 4, nf // nf, kernel_size=(5, 5, 5), stride=(2, 2, 2),
#         #     padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
#         # )
#         self.conv1 = self.ConvBlock(
#             nc_in, nf * 1, kernel_size=(3, 3, 3), stride=(2, 2, 2),
#             padding=1, bias=use_bias, norm=norm, conv_by=conv_by
#         )
#         # receptive field 1+(3-1)=3
#         self.conv2 = self.ConvBlock(
#             nf * 1, nf * 2, kernel_size=(3, 3, 3), stride=(2, 2, 2),
#             padding=(1, 1, 1), bias=use_bias, norm=norm, conv_by=conv_by
#         )
#         # receptive field 3+(3-1)*2=7
#         self.conv3 = self.ConvBlock(
#             nf * 2, nf * 4, kernel_size=(3, 3, 3), stride=(2, 2, 2),
#             padding=(1, 1, 1), bias=use_bias, norm=norm, conv_by=conv_by
#         )
#         # receptive field 7+(3-1)*2*2=15
#         self.conv4 = self.ConvBlock(
#             nf * 4, nf // nf, kernel_size=(3, 3, 3), stride=(2, 2, 2),
#             padding=(1, 1, 1), bias=use_bias, norm=norm, conv_by=conv_by
#         )
#         # receptive field 15+(3-1)*2*2*2=31
#         # self.conv5 = self.ConvBlock(
#         #     nf * 4, nf * 4, kernel_size=(5, 5, 5), stride=(2, 2, 2),
#         #     padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
#         # )
#         # self.conv6 = self.ConvBlock(
#         #     nf * 4, nf // nf, kernel_size=(5, 5, 5), stride=(2, 2, 2),
#         #     padding=(2, 2, 2), bias=use_bias, norm=None, activation=None,
#         #     conv_by=conv_by
#         # )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, xs):
#         c1 = self.conv1(xs)
#         c2 = self.conv2(c1)
#         c3 = self.conv3(c2)
#         c4 = self.conv4(c3)
#         # c5 = self.conv5(c4)
#         # c6 = self.conv6(c5)
#         if self.use_sigmoid:
#             c4 = torch.sigmoid(c4)
#         return c4

class SNTemporalPatchGANDiscriminator(BaseModule):
    def __init__(
            self, nc_in=2, nf=64, norm='SN', use_sigmoid=True, use_bias=True, conv_type='vanilla',
            conv_by='3d'
    ):
        super().__init__(conv_type)
        use_bias = use_bias
        self.use_sigmoid = use_sigmoid

        ######################
        # Convolution layers #
        ######################
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(5, 5, 5), stride=(2, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(5, 5, 5), stride=(2, 2, 2),
            padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(5, 5, 5), stride=(2, 2, 2),
            padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(5, 5, 5), stride=(2, 2, 2),
            padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by
        )
        self.conv5 = self.ConvBlock(
            nf * 4, 1, kernel_size=(5, 5, 5), stride=(2, 2, 2),
            padding=(2, 2, 2), bias=use_bias, norm=None, activation=None, conv_by=conv_by
        )
        # self.conv6 = self.ConvBlock(
        #     nf * 4, nf * 4, kernel_size=(5, 5, 5), stride=(2, 2, 2),
        #     padding=(2, 2, 2), bias=use_bias, norm=None, activation=None,
        #     conv_by=conv_by
        # )
        self.sigmoid = nn.Sigmoid()

    def forward(self, xs):
        c1 = self.conv1(xs)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        # c6 = self.conv6(c5)
        if self.use_sigmoid:
            c5 = torch.sigmoid(c5)
        return c5
