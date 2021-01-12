from models.unet import Kumar_Net, Dakai_Net, UNet
from models.discriminator import Kumar_discriminator, discriminator_v1, discriminator_v3, \
    SNTemporalPatchGANDiscriminator, Dakai_discriminator
from models.FFInpainting import *
from models.modelszoo.Densenet3D import SinglePathDenseNet, DualPathDenseNet, DualSingleDenseNet
from models.modelszoo.DenseVoxelNet import DenseVoxelNet
from models.modelszoo.HighResNet3D import HighResNet3D
from models.modelszoo.HyperDensenet import HyperDenseNet
from models.modelszoo.ResNet3DMedNet import generate_resnet3d
from models.modelszoo.SkipDenseNet3D import SkipDenseNet3D
from models.modelszoo.u2net import U2NETP
from models.modelszoo.unetfamily import AttU_Net

MODELS = {'Kumar': Kumar_Net,
          # 'raunet': RAUnet,
          'Kumar_discriminator': Kumar_discriminator,
          'Dakai': Dakai_Net,
          'Dakai_discriminator': Dakai_discriminator,

          'PatchGAN': SNTemporalPatchGANDiscriminator,
          'PatchGANDiscriminator': SNTemporalPatchGANDiscriminator,
          'noneofall': NoGateNoDilateNoRicherGenerator,
          'dilate': NoGateDilateNoRicherGenerator,
          'gatedilate': GateDilateNoRicherGenerator,
          'gatedilatericher': GateDilateRicherGenerator,
          'unet': UNet,
          'u2net': U2NETP,
          'MedNet3D': generate_resnet3d,
          'AttU_Net': AttU_Net,
          }
