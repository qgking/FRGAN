import functools
import torch
import torch.nn as nn
from os.path import isdir, join
import os
from utils.utils import visualize
from models.coarsenet import CoarseNet
from models.refinenet import RefineNet


class BaseGenerator(nn.Module):
    def __init__(
            self, nc_in, nc_out, nf=64, use_bias=True, norm='BN', conv_by='3d', conv_type='gated', upsample='noricher',
            use_refine=False, refine='Enhance'
    ):
        super().__init__()
        self.coarse_net = CoarseNet(
            nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, upsample
        )
        self.use_refine = use_refine
        if self.use_refine:
            self.refine_net = RefineNet(
                nc_in, nc_out, nf, use_bias, norm, conv_by, conv_type, upsample_type=refine
            )

    def preprocess(self, masked_imgs, masks):
        # B, L, C, H, W = masked.shape
        inp = torch.cat([masked_imgs, masks], dim=1)
        return inp

    def forward(self, masked_imgs, masks, pre_trained=False):
        if pre_trained:
            self.coarse_net.eval()
            with torch.no_grad():
                coarse_outputs, so_out_c, encoded_features, up_d = self.coarse_net(masked_imgs, masks)
        else:
            coarse_outputs, so_out_c, encoded_features, up_d = self.coarse_net(masked_imgs, masks)
        if self.use_refine:
            refined_outputs, so_out = self.refine_net(coarse_outputs, encoded_features, up_d, masks)
            return {
                "outputs": refined_outputs,
                "coarse_outputs": coarse_outputs,
                "so_out": so_out,
                "so_out_c": so_out_c
            }
        else:
            return {"outputs": coarse_outputs,
                    "so_out": so_out_c,
                    "coarse_outputs": None,
                    "so_out_c": None
                    }


#
class NoGateNoDilateNoRicherGenerator(BaseGenerator):
    def __init__(
            self, nc_in, nc_out, nf=64, use_bias=True, norm='BN', conv_by='3d', conv_type='nodilate',
            upsample='noricher',
            use_refine=False, refine='Refine'
    ):
        super(NoGateNoDilateNoRicherGenerator, self).__init__(nc_in, nc_out, nf=64,
                                                              use_bias=True,
                                                              norm='BN',
                                                              conv_by='3d',
                                                              conv_type='nodilate',
                                                              upsample='noricher',
                                                              use_refine=False)


# dilate
class NoGateDilateNoRicherGenerator(BaseGenerator):
    def __init__(
            self, nc_in, nc_out, nf=64, use_bias=True, norm='BN', conv_by='3d', conv_type='vanilla',
            upsample='noricher',
            use_refine=False, refine='RefineContext'
    ):
        super(NoGateDilateNoRicherGenerator, self).__init__(nc_in, nc_out, nf=64,
                                                            use_bias=True,
                                                            norm='BN',
                                                            conv_by='3d',
                                                            conv_type='vanilla',
                                                            upsample='noricher',
                                                            use_refine=False)


# gate dilate
class GateDilateNoRicherGenerator(BaseGenerator):
    def __init__(
            self, nc_in, nc_out, nf=64, use_bias=True, norm='BN', conv_by='3d', conv_type='gated', upsample='noricher',
            use_refine=False, refine='RefineContext'
    ):
        super(GateDilateNoRicherGenerator, self).__init__(nc_in, nc_out, nf=64,
                                                          use_bias=True,
                                                          norm='BN',
                                                          conv_by='3d',
                                                          conv_type='gated',
                                                          upsample='noricher',
                                                          use_refine=False)


# gate dilate richer
class GateDilateRicherGenerator(BaseGenerator):
    def __init__(
            self, nc_in, nc_out, nf=64, use_bias=True, norm='BN', conv_by='3d', conv_type='gated', upsample='richer',
            use_refine=False, refine='RefineContext'
    ):
        super(GateDilateRicherGenerator, self).__init__(nc_in, nc_out, nf=64,
                                                        use_bias=True,
                                                        norm='BN',
                                                        conv_by='3d',
                                                        conv_type='gated',
                                                        upsample='richer',
                                                        use_refine=False)

