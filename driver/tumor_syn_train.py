import matplotlib
import sys

sys.path.extend(["../", "./"])
from utils.utils import *
from os.path import join, isdir
import time
from models import MODELS
from driver.helper import HELPER
import torch
from torch.utils.data import DataLoader
from torch.cuda import empty_cache
from tumor_data.SYNDataLoader import SYNDataLoader
import random
from driver.Optim import Optimizer
from driver.Config import Configurable
from module.Critierion import L1LossMaskedMean, VGGLoss, StyleLoss
from models.vgg import Vgg16
import shutil
from skimage import measure, morphology
import torch.nn.functional as F
from driver.syn_train_test_utils import train, valid, test_old

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse


def main(config):
    # device = torch.device("cuda")
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

    syn = HELPER[helper](generator, discriminator,
                         criterion, config)
    # if 'allGateDilateRicherContextRefine' == syn.config.generator:
    #     optimizer_G = Optimizer(name=syn.config.learning_algorithm,
    #                             model=syn.generator.refine_net,
    #                             lr=syn.config.learning_rate)
    # else:
    optimizer_G = Optimizer(name=syn.config.learning_algorithm,
                            model=syn.generator,
                            lr=syn.config.learning_rate)
    optimizer_D = Optimizer(name=syn.config.learning_algorithm,
                            model=syn.discriminator,
                            lr=syn.config.learning_rate)
    syn.move_to_cuda()
    model_optimizer = {
        'G': optimizer_G,
        'D': optimizer_D,
    }
    # if 'allGateDilateRicherContextRefine' == syn.config.generator:
    #     syn.load_pretrained_coarse_weight()
    epo = syn.load_lastest_weight(model_optimizer)
    train_dataset = SYNDataLoader(root=syn.config.data_root, split_radio=syn.config.split_radio, split='train',
                                  data_type="SYN", config=syn.config)
    vali_dataset = SYNDataLoader(root=syn.config.data_root, split_radio=syn.config.split_radio, split='vali',
                                 data_type="SYN", config=syn.config)
    test_dataset = SYNDataLoader(root=syn.config.data_root, split_radio=syn.config.split_radio, split='test',
                                 data_type="SYN", config=syn.config)
    train_loader = DataLoader(train_dataset, batch_size=syn.config.train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=syn.config.workers)
    vali_loader = DataLoader(vali_dataset, batch_size=syn.config.train_batch_size, shuffle=False, pin_memory=True,
                             num_workers=syn.config.workers)
    # test_loader = DataLoader(test_dataset, batch_size=syn.config.test_batch_size, shuffle=False, pin_memory=True,
    #                          num_workers=syn.config.workers)

    # decay_epoch = [60, 90, 120, 150, 180, 210, 240]
    decay_epoch = [60, 70, 80, 90, 120, 150]
    decay_next = (np.array(decay_epoch) - epo > 0).argmax(axis=0)
    decay_e = decay_epoch[decay_next]
    for epoch in range(epo, syn.config.epochs):
        # increase boundary loss factor
        syn.boundary_loss_factor = min(
            syn.config.boundary_loss_factor / syn.config.max_boundary_loss_factor_epoch * epoch,
            syn.config.boundary_loss_factor)
        print("\nIncrease the boundary loss factor to %.3f" % (syn.boundary_loss_factor))
        train_critics = train(syn, train_loader, model_optimizer, epoch)
        syn.write_summary(epoch, train_critics)
        syn.plot_train_loss(epoch, train_critics)
        syn.save_model_checkpoint(epoch, model_optimizer)
        if (epoch + 1) % syn.config.validate_every_epoch == 0:
            vali_critics = valid(syn, vali_loader, epoch)
            syn.write_summary(epoch, vali_critics)
            syn.plot_vali_loss(epoch, vali_critics)
            # syn.write_summary(epoch, test_critics)
        # decay after start_decay_at
        # if (epoch + 1) % decay_e == 0:
        #     for g in model_optimizer['G'].param_groups:
        #         current_lr = max(g['lr'] * 0.5, syn.config.min_lrate)
        #         print("Decaying the learning ratio to %.8f" % (current_lr))
        #         g['lr'] = current_lr
        #     decay_next += 1
        #     decay_e = decay_epoch[decay_next]
        #     print("Next decay will be in the %d th epoch" % (decay_e))

    test_old(syn, test_dataset, epoch)
    syn.summary_writer.close()

    # test_dataset = SYNDataLoader(root='../data_root/patches/liver_tumor_syn_man_tumor_64_1', split='train',
    #                              data_type='SYN',
    #                              split_radio=1)
    # test_loader = DataLoader(test_dataset, batch_size=syn.config.test_batch_size, shuffle=False, pin_memory=True,
    #                          num_workers=syn.config.workers)
    # syn_test(syn, test_loader)


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
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=True)
    argparser.add_argument('--helper', type=str, default='none',
                           choices=['SYNHelper_Dakai', 'SYNHelper_Kumar', 'SYNHelper_None',
                                    'SYNHelper_Dilate_Gate_Richer',
                                    'SYNHelper_Dilate_Gate_Richer_Lcon',
                                    'SYNHelper_Dilate_Gate_Richer_Lcon_Lpercep',
                                    'SYNHelper_Dilate_Gate_Richer_Lcon_Lsty',
                                    'SYNHelper_Dilate_Gate_Richer_Lsty',
                                    'SYNHelper_Dilate_Gate_Richer_Lpercep',
                                    'SYNHelper_Dilate_Gate_Richer_Lcon_Lpercep_Lsty',
                                    'SYNHelper_Dilate_Gate_Lsty',
                                    'SYNHelper_Dilate_Gate_Lpercep',
                                    'SEGHelper'
                                    ])
    argparser.add_argument('--gpu', help='GPU 0,1,2,3', default=0)
    argparser.add_argument('--gpu-count', help='number of GPUs (0,1,2,3)', default='0')
    argparser.add_argument('--run-num', help='run num: 0,2,3', default=0)
    argparser.add_argument('--gen', type=str, default='none',
                           choices=['Kumar', 'Dakai', 'unet',
                                    # 'gateDilateRicherNoContextRefine',
                                    # 'NoGateDilateRicherContextRefine',
                                    # 'gateNoDilateRicherContextRefine',
                                    # 'gateDilateNoRicherContextRefine',
                                    'noneofall',
                                    # 'gatedilatericherRefine',
                                    'dilate',
                                    'gatedilate',
                                    # 'gatedilateNoricherNoSkipRefine',
                                    'gatedilatericher',
                                    # 'gateDilateNoRicherNoSkipRefineContext'
                                    ])

    argparser.add_argument('--dis', type=str, default='none',
                           choices=['Kumar_discriminator', 'Dakai_discriminator',
                                    'PatchGANDiscriminator', 'PatchGAN'])
    argparser.add_argument('--split', type=str, default='0.6')
    argparser.add_argument('--batch-size', type=str, default='4')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads((config.workers + 1) * len(config.gpu_count))

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
