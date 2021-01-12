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
from driver.helper.SEGHelper import SEGHelper
from driver.Config import Configurable
from torch.nn import BCELoss
from driver.syn_train_test_utils import test_seg_old
import numpy as np

import time

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse


def main(config):
    # Initialize generator and discriminator
    model = MODELS[config.seg_net](n_channels=1, n_classes=1)
    criterion = BCELoss()
    seg = SEGHelper(model, criterion,
                    config)

    weight_files = sorted(glob(join(seg.config.load_model_path, 'checkpoint_epoch_*.pth')), reverse=True)
    # weight_files = []
    # weight_files.append(join(TMP_DIR, 'checkpoint_epoch_006.pth'))
    print("loaded:" + weight_files[0])
    if not seg.config.use_cuda:
        seg.model.load_state_dict(torch.load(weight_files[0],
                                             map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                           'cuda:2': 'cpu', 'cuda:3': 'cpu'})['state_dict'])

    else:
        seg.model.load_state_dict(
            torch.load(weight_files[0], map_location=('cuda:' + str(seg.device)))['state_dict'])
    seg.move_to_cuda()

    test_dataset = SYNDataLoader(root=seg.config.data_root, split_radio=config.split_radio, split='test',
                                 data_type="SEG", config=seg.config)
    test_seg_old(seg, test_dataset, 'final_test_epoch')


if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='configuration.txt')
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
