import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


########################
# Convolutional Blocks #
########################

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 transpose=False, output_padding=0):
        super().__init__()
        if padding == -1:
            padding = ((np.array(kernel_size) - 1) * np.array(dilation)) // 2
            # to check if padding is not a 0-d array, otherwise tuple(padding) will raise an exception
            if hasattr(padding, '__iter__'):
                padding = tuple(padding)

        if transpose:
            self.conv = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size,
                stride, padding, output_padding, groups, bias, dilation)
        else:
            self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
                 transpose=False, output_padding=0):
        super().__init__()
        if padding == -1:
            padding = ((np.array(kernel_size) - 1) * np.array(dilation)) // 2
            if hasattr(padding, '__iter__'):
                padding = tuple(padding)

        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size,
                stride, padding, output_padding, groups, bias, dilation)
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride, padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.conv = nn.utils.spectral_norm(self.conv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.conv(xs)
        if self.activation is not None:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
            groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'
    ):

        super().__init__()
        if conv_by == '3d':
            self.module = torch.nn
        else:
            raise NotImplementedError(f'conv_by {conv_by} is not implemented.')

        self.padding = tuple(((np.array(kernel_size) - 1) * np.array(dilation)) // 2) if padding == -1 else padding
        self.featureConv = self.module.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = self.module.BatchNorm3d(out_channels)
        elif norm == "IN":
            self.norm_layer = self.module.InstanceNorm3d(out_channels, track_running_stats=True)
        elif norm == "SN":
            self.norm = None
            self.featureConv = nn.utils.spectral_norm(self.featureConv)
        elif norm is None:
            self.norm = None
        else:
            raise NotImplementedError(f"Norm type {norm} not implemented")

        self.activation = activation

    def forward(self, xs):
        out = self.featureConv(xs)
        if self.activation:
            out = self.activation(out)
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class VanillaDeconv(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
            groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
            scale_factor=2, conv_by='3d'
    ):
        super().__init__()
        self.conv = VanillaConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by=conv_by)
        self.scale_factor = scale_factor

    def forward(self, xs):
        xs_resized = F.interpolate(xs, scale_factor=(self.scale_factor, self.scale_factor, self.scale_factor))
        return self.conv(xs_resized)


class GatedConv(VanillaConv):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
            groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True), conv_by='3d'
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by
        )
        self.gatingConv = self.module.Conv3d(
            in_channels, out_channels, kernel_size,
            stride, self.padding, dilation, groups, bias)
        if norm == 'SN':
            self.gatingConv = nn.utils.spectral_norm(self.gatingConv)
        self.sigmoid = nn.Sigmoid()
        self.store_gated_values = False

    def gated(self, mask):
        # return torch.clamp(mask, -1, 1)
        out = self.sigmoid(mask)
        if self.store_gated_values:
            self.gated_values = out.detach().cpu()
        return out

    def forward(self, xs):
        gating = self.gatingConv(xs)
        feature = self.featureConv(xs)
        if self.activation:
            feature = self.activation(feature)
        out = self.gated(gating) * feature
        if self.norm is not None:
            out = self.norm_layer(out)
        return out


class GatedDeconv(VanillaDeconv):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
            groups=1, bias=True, norm="SN", activation=nn.LeakyReLU(0.2, inplace=True),
            scale_factor=2, conv_by='3d'
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, scale_factor, conv_by
        )
        self.conv = GatedConv(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, norm, activation, conv_by=conv_by)


class ContextualAttentionModule(nn.Module):

    def __init__(self, patch_size=3, propagate_size=3, stride=1):
        super(ContextualAttentionModule, self).__init__()
        self.patch_size = patch_size
        self.propagate_size = propagate_size
        self.stride = stride
        self.prop_kernels = None

    def forward(self, foreground, mask, background="same"):
        ###assume the masked area has value 1
        bz, nc, w, h, d = foreground.size()
        if background == "same":
            background = foreground.clone()
        background = background * (1 - mask)
        background = F.pad(background,
                           [self.patch_size // 2, self.patch_size // 2, self.patch_size // 2, self.patch_size // 2,
                            self.patch_size // 2, self.patch_size // 2])
        conv_kernels_all = background.unfold(2, self.patch_size, self.stride) \
            .unfold(3, self.patch_size, self.stride) \
            .unfold(4, self.patch_size, self.stride) \
            .contiguous().view(bz, nc, -1, self.patch_size, self.patch_size, self.patch_size)
        conv_kernels_all = conv_kernels_all.transpose(2, 1)
        output_tensor = []
        for i in range(bz):
            feature_map = foreground[i:i + 1]

            # form convolutional kernels
            conv_kernels = conv_kernels_all[i] + 0.0000001
            norm_factor = torch.sum(conv_kernels ** 2, [1, 2, 3, 4], keepdim=True) ** 0.5
            conv_kernels = conv_kernels / norm_factor

            conv_result = F.conv3d(feature_map, conv_kernels, padding=self.patch_size // 2)
            if self.propagate_size != 1:
                if self.prop_kernels is None:
                    self.prop_kernels = torch.ones(
                        [conv_result.size(1), 1, self.propagate_size, self.propagate_size, self.propagate_size])
                    self.prop_kernels.requires_grad = False
                    self.prop_kernels = self.prop_kernels.cuda()
                conv_result = F.conv3d(conv_result, self.prop_kernels, stride=1, padding=1, groups=conv_result.size(1))
            attention_scores = F.softmax(conv_result, dim=1)
            ##propagate the scores
            recovered_foreground = F.conv_transpose3d(attention_scores, conv_kernels, stride=1,
                                                      padding=self.patch_size // 2)
            # average the recovered value, at the same time make non-masked area 0
            recovered_foreground = (recovered_foreground * mask[i]) / (self.patch_size ** 3)
            # recover the image
            final_output = recovered_foreground + feature_map * (1 - mask[i])
            output_tensor.append(final_output)
        return torch.cat(output_tensor, dim=0)
