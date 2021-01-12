import sys

import matplotlib

sys.path.extend(["../", "./"])
from utils.utils import *
import time
from models import MODELS
import torch
from torch.utils.data import DataLoader
from torch.cuda import empty_cache
from tumor_data.SYNDataLoader import SYNDataLoader, SEGDataLoader
import random
from driver.Optim import Optimizer
from driver.helper.SEGHelper import SEGHelper
from driver.Config import Configurable
from module.Critierion import BCELoss
from driver.syn_train_test_utils import test_seg_old

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse

log_template = "\n[Epoch %d/%d] [Batch %d/%d] [printfreq_time ETA: %.3f] [seg loss: %f] [seg acc: %f]  ETA: %.3f"


def main(config, args):
    # Initialize generator and discriminator
    model = MODELS[config.seg_net](n_channels=1, n_classes=1)
    criterion = BCELoss()
    seg = SEGHelper(model, criterion,
                    config)
    seg.move_to_cuda()
    optimizer = Optimizer(name=seg.config.learning_algorithm,
                          model=seg.model,
                          lr=seg.config.learning_rate)
    train_dataset = SYNDataLoader(root=seg.config.data_root, split_radio=config.split_radio, split='train',
                                  data_type="SYN", config=seg.config)
    train_patches_imgs = train_dataset.patches_imgs
    if 'none' != args.syn_dataset:
        if 'aug_normal' in args.syn_dataset:
            syn_data_set = "../data_root/patches/" + args.syn_dataset
            syn_datalist = sorted(glob(join(syn_data_set, "*.npy")),
                                  reverse=False)
            print(args.syn_dataset)
            print("dataset 1:" + seg.config.data_root)
            print("dataset 2:" + syn_data_set)
            print("train_patches_imgs %s syn_datalist %s:" % (len(train_patches_imgs), len(syn_datalist)))
            # train_patches_imgs = train_patches_imgs + syn_datalist
            train_patches_imgs =  syn_datalist
        else:
            valid_dataset = SYNDataLoader(root=seg.config.data_root, split_radio=config.split_radio, split='vali',
                                          data_type="SEG", config=seg.config)
            syn_data_set = "../data_root/patches/" + seg.config.data_name + "/" + args.syn_dataset
            syn_datalist = sorted(glob(join(syn_data_set, "*.npy")),
                                  reverse=False)
            valid_imgs = sorted(valid_dataset.patches_imgs_idx, reverse=False)
            np.random.shuffle(valid_imgs)
            syn_idx = valid_imgs[:int(len(valid_imgs) * args.partition)]
            normal_idx = valid_imgs[int(len(valid_imgs) * args.partition):]
            st_idx = syn_datalist[0].rfind('img_') + 4
            end_edx = st_idx + 3
            syn_patches_imgs = [path for path in syn_datalist if path[st_idx:end_edx] in syn_idx]
            st_idx = valid_dataset.patches_imgs[0].rfind('img_') + 4
            end_edx = st_idx + 3
            normal_patches_imgs = [path for path in valid_dataset.patches_imgs if path[st_idx:end_edx] in normal_idx]
            print(args.syn_dataset)
            print("dataset 1:" + seg.config.data_root)
            print("dataset 2:" + syn_data_set)
            print("syn_idx %s syn_patches_imgs %s:" % (len(syn_idx), len(syn_patches_imgs)))
            print(syn_idx)
            print("normal_idx %s normal_patches_imgs %s:" % (len(normal_idx), len(normal_patches_imgs)))
            print(normal_idx)
            # img_file = sorted(syn_patches_imgs)[0]
            # img = np.load(join(img_file))
            # img_ori = torch.from_numpy(np.expand_dims(img[..., 0], 0)).unsqueeze(0)
            # seg_ = torch.from_numpy(np.expand_dims(img[..., 4], 0)).unsqueeze(0)
            # base_name = os.path.basename(img_file)[:-4]
            # visual_batch(img_ori, config.tmp_dir, "%s_syn_A" % (base_name), channel=1, nrow=8)
            # visual_batch(seg_, config.tmp_dir, "%s_seg_A" % (base_name), channel=1, nrow=8)
            train_patches_imgs = syn_patches_imgs + normal_patches_imgs
            # train_patches_imgs = train_patches_imgs + syn_patches_imgs + normal_patches_imgs

    train_dataset = SEGDataLoader(train_patches_imgs, config,
                                  config.split_radio)
    train_loader = DataLoader(train_dataset, batch_size=seg.config.train_batch_size, shuffle=True, pin_memory=True,
                              num_workers=seg.config.workers)
    test_dataset = SYNDataLoader(root=seg.config.data_root, split_radio=seg.config.split_radio, split='test',
                                 data_type="SEG", config=seg.config)
    for epoch in range(seg.config.epochs):
        train(seg, train_loader, optimizer, epoch)
        opt_s = ''
        for g in optimizer.param_groups:
            opt_s += "optimizer current_lr to %.8f \t" % (g['lr'])
        print(opt_s)
        seg.save_model_checkpoint(epoch, optimizer)
        test_seg_old(seg, test_dataset, epoch)
        seg.log.flush()
    seg.summary_writer.close()


def train(seg, train_loader, optimizer, epoch):
    seg.model.train()
    end = time.time()
    pend = time.time()
    batch_time = Averagvalue()
    printfreq_time = Averagvalue()
    losses = Averagvalue()
    acces = Averagvalue()
    batch_num = int(np.ceil(len(train_loader.dataset) / float(seg.config.train_batch_size)))
    total_iter = batch_num * seg.config.epochs
    for i, batch in enumerate(train_loader):
        real_A, real_B = seg.generate_batch(batch)
        seg.adjust_learning_rate(optimizer, epoch * batch_num + i, total_iter)
        optimizer.zero_grad()
        loss_s, acc_s = seg.train_model_one_batch(
            real_A, real_B)
        losses.update(loss_s, real_A.size(0))
        acces.update(acc_s)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        # Print log
        if (i + 1) % (int(len(train_loader) / seg.config.printfreq)) == 0:
            printfreq_time.update(time.time() - pend)
            pend = time.time()
            print(
                log_template % (
                    epoch,
                    seg.config.epochs,
                    i,
                    len(train_loader),
                    printfreq_time.val,
                    losses.avg, acces.avg,
                    batch_time.val,
                )
            )
    empty_cache()
    return losses.avg


if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    torch.backends.cudnn.benchmark = True  # cudn
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='configuration.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='test not need write', default=True)
    argparser.add_argument('--syn-dataset', type=str, default='none',
                           choices=['Dakai_SYNHelper_Dakai',
                                    'Kumar_SYNHelper_Kumar',
                                    'noneofall_SYNHelper_None',
                                    'none',
                                    'gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lcon_Lpercep_Lsty',
                                    'gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lcon_Lpercep',
                                    'gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lcon',
                                    'gatedilatericher_SYNHelper_Dilate_Gate_Richer', 'gatedilate_SYNHelper_None',
                                    'dilate_SYNHelper_None',
                                    'gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lcon_Lsty',
                                    'gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lsty',
                                    'gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lpercep',
                                    'gatedilatericher_SYNHelper_Dilate_Gate_Lsty',
                                    'gatedilatericher_SYNHelper_Dilate_Gate_Lpercep',
                                    'kidney_tumor_syn_tumor_aug_normal',
                                    'liver_tumor_syn_tumor_aug_normal'
                                    ])
    argparser.add_argument('--gpu', help='GPU 0,1,2,3', default=0)
    argparser.add_argument('--gpu-count', help='number of GPUs (0,1,2,3)', default='0')
    argparser.add_argument('--run-num', help='run num: 0,2,3', default=0)
    argparser.add_argument('--split', type=str, default='0.6')
    argparser.add_argument('--seg', type=str, default='unet',
                           choices=['unet', 'u2net', 'MedNet3D', 'AttU_Net'])
    argparser.add_argument('--partition', type=float, default=1, choices=[0, 0.25, 0.5, 0.75, 1])
    # -----------------------
    argparser.add_argument('--batch-size', type=str, default='4')
    argparser.add_argument('--helper', type=str, default='none')
    argparser.add_argument('--gen', type=str, default='none')
    argparser.add_argument('--dis', type=str, default='none')

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    config.set_attr('Network', 'seg_net', args.seg)
    torch.set_num_threads(config.workers + 3)

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config, args)
