import torch
import torch.nn as nn
import torch.nn.functional as F
from module.blocks import (
    GatedConv, GatedDeconv,
    VanillaConv, VanillaDeconv
)
from module.blocks import ContextualAttentionModule


###########################
# Encoder/Decoder Modules #
###########################
def crop(variable, th, tw, td):
    h, w, d = variable.shape[2], variable.shape[3], variable.shape[4]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    z1 = int(round((d - td) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw, z1: z1 + td]


class BaseModule(nn.Module):
    def __init__(self, conv_type):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == 'gated' or conv_type == 'nodilate':
            self.ConvBlock = GatedConv
            self.DeconvBlock = GatedDeconv
        elif conv_type == 'vanilla':
            self.ConvBlock = VanillaConv
            self.DeconvBlock = VanillaDeconv

    def concat_feature(self, ca, cb):
        if self.conv_type == 'partial':
            ca_feature, ca_mask = ca
            cb_feature, cb_mask = cb
            feature_cat = torch.cat((ca_feature, cb_feature), 1)
            # leave only the later mask
            return feature_cat, ca_mask
        else:
            return torch.cat((ca, cb), 1)


class DownSampleModule(BaseModule):
    def __init__(self, nc_in, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(conv_type)
        self.conv_type = conv_type
        self.conv1 = self.ConvBlock(
            nc_in, nf * 1, kernel_size=(5, 5, 5), stride=1,
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Downsample 1
        self.conv2 = self.ConvBlock(
            nf * 1, nf * 2, kernel_size=(4, 4, 4), stride=(2, 2, 2),
            padding=(2, 2, 2), bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv3 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        # Downsample 2
        self.conv4 = self.ConvBlock(
            nf * 2, nf * 4, kernel_size=(4, 4, 4), stride=(2, 2, 2),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv5 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv6 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=norm, conv_by=conv_by)

        # Dilated Convolutions
        self.dilated_conv1 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(2, 2, 2))
        self.dilated_conv2 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(4, 4, 4))
        self.dilated_conv3 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(8, 8, 8))
        self.dilated_conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=-1, bias=use_bias, norm=norm, conv_by=conv_by, dilation=(16, 16, 16))

        # No Dilated Convolutions
        self.normal_conv1 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.normal_conv2 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.normal_conv3 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.normal_conv4 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        self.conv7 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv8 = self.ConvBlock(
            nf * 4, nf * 4, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

    def forward(self, inp):
        c1 = self.conv1(inp)
        c2 = self.conv2(c1)

        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        if 'nodilate' not in self.conv_type:
            # dilate conv
            a1 = self.dilated_conv1(c6)
            a2 = self.dilated_conv2(a1)
            # a3 = self.dilated_conv3(a2)
            # a4 = self.dilated_conv4(a3)
        else:
            # normal conv
            a1 = self.normal_conv1(c6)
            a2 = self.normal_conv2(a1)
            # a3 = self.normal_conv3(a2)
            # a4 = self.normal_conv4(a3)

        c7 = self.conv7(a2)
        c8 = self.conv8(c7)
        return c8, c7, c4, c2, a2  # For skip connection


class UpSampleModule(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, upsample_type):
        super().__init__(conv_type)
        self.upsample_type = upsample_type
        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nc_in, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        # Upsample 2
        # self.deconv2 = self.DeconvBlock(
        #     nf * 2, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
        #     bias=use_bias, norm=norm, conv_by=conv_by)

        self.deconv2 = self.DeconvBlock(
            nf * 4, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        self.conv10 = self.ConvBlock(
            nf * 1, nf * 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        # self.context_att = ContextualAttentionModule(3)
        self.conv11 = self.ConvBlock(
            nf * 1, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)

        self.score_dsn1 = nn.Conv3d(nf * 4, 1, kernel_size=1)
        self.score_dsn2 = nn.Conv3d(nf * 2, 1, kernel_size=1)
        self.score_dsn3 = nn.Conv3d(nf * 1, 1, kernel_size=1)
        self.score_final = nn.Conv3d(3, 1, 1)

    def forward(self, inp, up_d, coarse_output, mask):
        c8, c7, c4, c2, a = inp
        # c8, c7, c4, c2, input = inp
        # img_H, img_W, img_D = input.shape[2], input.shape[3], input.shape[4]
        concat1 = self.concat_feature(c8, c4)
        d1 = self.deconv1(concat1)
        c9 = self.conv9(d1)
        # retrain for this part beacause its missing
        concat2 = self.concat_feature(c9, c2)
        d2 = self.deconv2(concat2)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        so_out_final = None
        if 'noricher' not in self.upsample_type:
            # richer conv branch
            so1_out = self.score_dsn1(c7 + c8)
            so2_out = self.score_dsn2(d1 + c9)
            so3_out = self.score_dsn3(d2 + c10)

            upsample1 = F.interpolate(so1_out, scale_factor=4, mode="trilinear", align_corners=True)
            upsample2 = F.interpolate(so2_out, scale_factor=2, mode="trilinear", align_corners=True)

            # so3_out = crop(so3_out, img_H, img_W, img_D)
            # upsample2 = crop(upsample2, img_H, img_W, img_D)
            # upsample1 = crop(upsample1, img_H, img_W, img_D)
            fusecat = torch.cat((so3_out, upsample2, upsample1), dim=1)
            fuse = self.score_final(fusecat)
            so_out_final = [so3_out, upsample2, upsample1, fuse]
            # so_out_final = [torch.sigmoid(r) for r in so_out_final]
        return c11, so_out_final, [d1, d2]


class NoRicherNoSkipUpSampleModule(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type):
        super().__init__(conv_type)
        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nc_in, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        # Upsample 2
        # self.deconv2 = self.DeconvBlock(
        #     nf * 2, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
        #     bias=use_bias, norm=norm, conv_by=conv_by)

        self.deconv2 = self.DeconvBlock(
            nf * 2, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)

        self.conv10 = self.ConvBlock(
            nf * 1, nf * 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv11 = self.ConvBlock(
            nf * 1, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)

    def forward(self, inp, up_d, coarse_output, mask):
        c8, c7, c4, c2, a = inp
        # c8, c7, c4, c2, input = inp
        # img_H, img_W, img_D = input.shape[2], input.shape[3], input.shape[4]
        d1 = self.deconv1(c8)
        c9 = self.conv9(d1)
        # retrain for this part beacause its missing
        d2 = self.deconv2(c9)
        # d2 = self.deconv2(c9)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        return c11, None, [d1, d2]


class UpSampleRefineContextModule(BaseModule):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, upsample_type):
        super().__init__(conv_type)
        self.upsample_type = upsample_type
        self.context_att_1 = ContextualAttentionModule(3)
        # Upsample 1
        self.deconv1 = self.DeconvBlock(
            nf * 4 * 2, nf * 2, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv9 = self.ConvBlock(
            nf * 2, nf * 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        # Upsample 2
        # self.deconv2 = self.DeconvBlock(
        #     nf * 2, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
        #     bias=use_bias, norm=norm, conv_by=conv_by)

        self.deconv2 = self.DeconvBlock(
            nf * 2 * 2, nf * 1, kernel_size=(3, 3, 3), stride=1, padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.conv10 = self.ConvBlock(
            nf * 1, nf * 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1,
            bias=use_bias, norm=norm, conv_by=conv_by)
        self.context_att_2 = ContextualAttentionModule(3)
        self.conv11 = self.ConvBlock(
            nf * 1, nc_out, kernel_size=(3, 3, 3), stride=(1, 1, 1),
            padding=1, bias=use_bias, norm=None, activation=None, conv_by=conv_by)

        self.score_dsn1 = nn.Conv3d(nf * 4, 1, kernel_size=1)
        self.score_dsn2 = nn.Conv3d(nf * 2, 1, kernel_size=1)
        self.score_dsn3 = nn.Conv3d(nf * 1, 1, kernel_size=1)
        self.score_final = nn.Conv3d(3, 1, 1)

    def forward(self, inp, d_up, coarse_output, masks):
        c8, c7, c4, c2, a = inp
        # c7_scale_coarse = F.interpolate(coarse_output, size=c7.size()[2:])
        if 'NoContext' not in self.upsample_type:
            mask = F.interpolate(masks, size=c4.size()[2:])
            ctx_att_1 = self.context_att_1(c4, mask)
            concat1 = torch.cat((c8, ctx_att_1), 1)
        else:
            concat1 = torch.cat((c8, c4), 1)

        d1 = self.deconv1(concat1)
        c9 = self.conv9(d1)
        # if 'NoContext' not in self.upsample_type:
        #     mask = F.interpolate(masks, size=c2.size()[2:])
        #     ctx_att_2 = self.context_att_2(c2, mask)
        #     concat2 = torch.cat((c9, ctx_att_2), 1)
        # else:
        #     concat2 = torch.cat((c9, c2), 1)
        # d1_scale_coarse = F.interpolate(coarse_output, size=d1.size()[2:])
        concat2 = torch.cat((c9, c2), 1)

        # retrain for this part beacause its missing
        d2 = self.deconv2(concat2)
        c10 = self.conv10(d2)
        c11 = self.conv11(c10)
        so_out_final = None
        if 'noricher' not in self.upsample_type:
            so1_out = self.score_dsn1(c7 + c8)
            so2_out = self.score_dsn2(d1 + c9)
            so3_out = self.score_dsn3(d2 + c10)

            upsample1 = F.interpolate(so1_out, size=coarse_output.size()[2:], mode="trilinear", align_corners=True)
            upsample2 = F.interpolate(so2_out, size=coarse_output.size()[2:], mode="trilinear", align_corners=True)

            fusecat = torch.cat((so3_out, upsample2, upsample1), dim=1)
            fuse = self.score_final(fusecat)
            so_out_final = [so3_out, upsample2, upsample1, fuse]
        return c11, so_out_final, [d1, d2]
