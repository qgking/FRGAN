import matplotlib
import sys

sys.path.extend(["../", "./"])
from utils.utils import *
from os.path import join
import time
from models import MODELS
import torch
from tumor_data.SYNDataLoader import SYNDataLoader
import random
from driver.Config import Configurable
from driver.syn_train_test_utils import test_old
import numpy as np
from driver.helper import HELPER
from driver.Optim import Optimizer
from module.Critierion import L1LossMaskedMean, VGGLoss, StyleLoss
from models.vgg import Vgg16
matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse


def main(config):
    # Initialize generator and discriminator
    helper = config.helper
    print("helper:\t" + helper)
    if 'Kumar' in helper or 'Dakai' in helper:
        generator = MODELS[config.generator](n_channels=1, n_classes=1)
        discriminator = MODELS[config.discriminator](in_channels=2)
    else:
        # Initialize generator and discriminator
        generator = MODELS[config.generator](nc_in=2, nc_out=1)
        discriminator = MODELS[config.discriminator](nc_in=2)

    # criterion_GAN = torch.nn.L1Loss()
    if 'Lcon' in helper:
        vgg = Vgg16(requires_grad=False)
        if config.use_cuda:
            device = config.gpu if config.use_cuda else None
            torch.cuda.set_device(device)
            vgg.cuda()
        criterion = {
            'criterion_GAN': torch.nn.MSELoss(),
            'criterion_pixelwise': torch.nn.L1Loss(),
            'L1LossMaskedMean': L1LossMaskedMean(),
            'VGGLoss': VGGLoss(vgg),
            'StyleLoss': StyleLoss(vgg)
        }
    else:
        criterion = {
            'criterion_GAN': torch.nn.MSELoss(),
            'criterion_pixelwise': torch.nn.L1Loss(),
            'L1LossMaskedMean': L1LossMaskedMean(),
        }

    syn = HELPER[helper](generator, discriminator,
                         criterion, config)
    syn.move_to_cuda()
    weight_files = sorted(glob(join(syn.config.load_model_path, 'checkpoint_epoch_*.pth')), reverse=True)
    if len(weight_files) > 0:
        syn.load_generator_history_weight(weight_files[0])
    else:
        exit(0)
    test_dataset = SYNDataLoader(root=syn.config.data_root, split_radio=syn.config.split_radio, split='test',
                                 data_type="SYN", config=syn.config)
    test_old(syn, test_dataset, 'final_test_epoch')


if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='syn_configuration.txt')
    argparser.add_argument('--thread', default=1, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=False)
    argparser.add_argument('--gpu', help='GPU 0,1,2,3', default=0)
    argparser.add_argument('--gpu-count', help='number of GPUs (0,1,2,3)', default='0')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads(args.thread)

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
