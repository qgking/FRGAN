# -*- coding: utf-8 -*-
# @Time    : 20/1/3 9:23
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : coarsenet.py

import functools
import torch
import torch.nn as nn
from os.path import isdir, join
import os
from module.modules import *
from utils.utils import visualize
import numpy as np


##################
# Generators #
##################

class CoarseNet(nn.Module):

    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, upsample_type):
        super().__init__()
        self.conv_type = conv_type
        self.upsample_type = upsample_type
        self.downsample_module = DownSampleModule(
            nc_in, nf, use_bias, norm, conv_by, conv_type)
        if self.upsample_type == 'norichernoskip':
            self.upsample_module = NoRicherNoSkipUpSampleModule(
                nf * 4, nc_out, nf, use_bias, norm, conv_by, conv_type)
        else:
            self.upsample_module = UpSampleModule(
                nf * 4 * 2, nc_out, nf, use_bias, norm, conv_by, conv_type, upsample_type=self.upsample_type)

    def preprocess(self, masked_imgs, masks):
        # B, L, C, H, W = masked.shape
        if self.conv_type == 'gated' or self.conv_type == 'vanilla' or self.conv_type == 'nodilate':
            inp = torch.cat([masked_imgs, masks], dim=1)
        else:
            raise NotImplementedError(f"{self.conv_type} not implemented")

        return inp

    def postprocess(self, masked_imgs, masks, c11):
        if self.conv_type == 'partial':
            inpainted = c11[0] * masks
        else:
            inpainted = c11 * masks

        out = inpainted + masked_imgs * (1 - masks)
        # msk_imgs = masked_imgs.detach().cpu().numpy()
        # masksss = masks.detach().cpu().numpy()
        # c11c11 = c11.detach().cpu().numpy()
        # inpaintedinpainted = inpainted.detach().cpu().numpy()
        # outout = out.detach().cpu().numpy()
        #
        # for b in range(msk_imgs.shape[0]):
        #     save_dir = './tmp/epoch/batch_' + str(b) + '/'
        #     if not isdir(save_dir):
        #         os.makedirs(save_dir)
        #     fake_B_numpy = msk_imgs[b]
        #     real_A_numpy = masksss[b]
        #     fake_C_numpy = c11c11[b]
        #     real_D_numpy = inpaintedinpainted[b]
        #     real_E_numpy = outout[b]
        #     for sss in range(fake_B_numpy.shape[-1]):
        #         visualize(np.expand_dims(fake_B_numpy[0, ..., sss], -1), join(save_dir, str(sss) + "Amasked_imgs"))
        #         visualize(np.expand_dims(real_A_numpy[0, ..., sss], -1), join(save_dir, str(sss) + "Bmasks"))
        #         visualize(np.expand_dims(fake_C_numpy[0, ..., sss], -1), join(save_dir, str(sss) + "Cc11"))
        #         visualize(np.expand_dims(real_D_numpy[0, ..., sss], -1), join(save_dir, str(sss) + "Dinpainted"))
        #         visualize(np.expand_dims(real_E_numpy[0, ..., sss], -1), join(save_dir, str(sss) + "Eout"))
        return out

    def forward(self, masked_imgs, masks):
        # B, L, C, H, W = masked.shape
        inp = self.preprocess(masked_imgs, masks)

        encoded_features = self.downsample_module(inp)

        c11, so_out_final, up_d = self.upsample_module(encoded_features, None, None, masks)

        out = self.postprocess(masked_imgs, masks, c11)

        return out, so_out_final, encoded_features, up_d
