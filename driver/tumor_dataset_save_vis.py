import matplotlib
import sys

sys.path.extend(["../", "./"])
from utils.utils import *
from os.path import join
import torch
from torch.utils.data import DataLoader
from tumor_data.SYNDataLoader import SYNDataLoader
import random
from driver.Config import Configurable
from os.path import join, isdir

import numpy as np

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse


def main():
    # Initialize generator and discriminator
    root = '../../patches/LiTS17/gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lcon_Lpercep_Lsty'
    test_dataset = SYNDataLoader(
        root=root,
        split_radio=1, split='train',
        data_type="SEG")
    print(root)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, pin_memory=True,
                             num_workers=4)
    for i, batch in enumerate(test_loader):
        img_erase = batch['img_erase'].detach().cpu().numpy()
        img_ori = batch['img_ori'].detach().cpu().numpy()
        img_seg = batch['img_seg'].detach().cpu().numpy()
        img_erase = np.squeeze(img_erase, 1)
        img_ori = np.squeeze(img_ori, 1)
        img_seg = np.squeeze(img_seg, 1)
        print('batch %d' % (i))
        for b in range(img_erase.shape[0]):
            img_erase_ = img_erase[b]
            img_ori_ = img_ori[b]
            img_seg_ = img_seg[b]
            save_dir = root + '/test/test_batch_' + str(i) + '/' + str(b) + '/'
            if not isdir(save_dir):
                os.makedirs(save_dir)
            for sss in range(img_erase_.shape[-1]):
                visualize(np.expand_dims(img_erase_[..., sss], -1),
                          join(save_dir,
                               '{0:0>3}'.format(i) + '{0:0>2}'.format(b) + '{0:0>2}'.format(sss) + "Aimg_erase"))
                visualize(np.expand_dims(img_ori_[..., sss], -1),
                          join(save_dir,
                               '{0:0>3}'.format(i) + '{0:0>2}'.format(b) + '{0:0>2}'.format(sss) + "Bimg_ori"))
                visualize(np.expand_dims(img_seg_[..., sss], -1),
                          join(save_dir,
                               '{0:0>3}'.format(i) + '{0:0>2}'.format(b) + '{0:0>2}'.format(sss) + "Cimg_seg"))


if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--thread', default=1, type=int, help='thread num')

    args, extra_args = argparser.parse_known_args()
    torch.set_num_threads(args.thread)
    main()
