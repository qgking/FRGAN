# -*- coding: utf-8 -*-
# @Time    : 20/1/3 9:23
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : refinenet.py

import functools
import torch
import torch.nn as nn
from os.path import isdir, join
import os
from module.modules import *
from utils.utils import visualize
from models.coarsenet import CoarseNet
import numpy as np
from module.modules import *
from module.blocks import (
    GatedConv, GatedDeconv,
    VanillaConv, VanillaDeconv
)


class RefineNet(CoarseNet):
    def __init__(self, nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, upsample_type):
        super().__init__(nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, upsample_type)
        self.conv_type = conv_type
        self.upsample_type = upsample_type
        self.downsample_module = DownSampleModule(
            nc_in, nf, use_bias, norm, conv_by, conv_type)
        self.upsample_module = UpSampleRefineContextModule(
            nf * 4 * 2, nc_out, nf, use_bias, norm, conv_by, conv_type, upsample_type=self.upsample_type)

    def forward(self, coarse_output, encoded_features_, up_d, masks):
        inp = self.preprocess(coarse_output, masks)
        encoded_features = self.downsample_module(inp)
        c11, so_out_final, _ = self.upsample_module(encoded_features, None, coarse_output, masks)
        out = self.postprocess(coarse_output, masks, c11)
        return out, so_out_final
