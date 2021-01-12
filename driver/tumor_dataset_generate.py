import matplotlib
import sys

sys.path.extend(["../", "./"])
from utils.utils import *
from os.path import join, isdir
from models import MODELS
import torch
from tumor_data.SYNDataLoader import SYNDataLoader
import random
from driver.helper import HELPER
from driver.Config import Configurable
from module.Critierion import L1LossMaskedMean, VGGLoss, StyleLoss, L2LossMaskedMean
from models.vgg import Vgg16
import shutil

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse


def main(config):
    # Initialize generator and discriminator
    helper = config.helper
    if 'Kumar' in helper or 'Dakai' in helper:
        generator = MODELS[config.generator](n_channels=1, n_classes=1)
        discriminator = MODELS[config.discriminator](in_channels=2)
    else:
        # Initialize generator and discriminator
        generator = MODELS[config.generator](nc_in=2, nc_out=1)
        discriminator = MODELS[config.discriminator](nc_in=2)

    vgg = Vgg16(requires_grad=False)
    if config.use_cuda:
        device = config.gpu if config.use_cuda else None
        torch.cuda.set_device(device)
        vgg.cuda()
    criterion = {
        'criterion_GAN': torch.nn.L1Loss(),
        'criterion_pixelwise': torch.nn.L1Loss(),
        'L1LossMaskedMean': L1LossMaskedMean(),
        'VGGLoss': VGGLoss(vgg),
        'StyleLoss': StyleLoss(vgg),
        "L2LossMaskedMean": L2LossMaskedMean()
    }

    print("helper:\t" + helper)
    syn = HELPER[helper](generator, discriminator,
                         criterion, config)
    # weight_files = sorted(glob(join(
    #     '../log/' + syn.config.data_name + '/' + syn.config.generator + '_' + syn.config.helper + '/' + syn.config.discriminator + '_' + str(
    #         syn.config.gpu) + '/checkpoint',
    #     'checkpoint_epoch_*.pth')), reverse=True)
    weight_files = sorted(glob(join(syn.config.load_model_path, 'checkpoint_epoch_*.pth')), reverse=True)

    print("loaded:" + weight_files[0])
    syn.load_generator_history_weight(weight_files[0])
    syn.move_to_cuda()
    vali_dataset = SYNDataLoader(root=syn.config.data_root, split_radio=config.split_radio, split='vali',
                                 config=syn.config, data_type="SYN")
    test(syn, vali_dataset)
    exit(0)


def test(syn, vali_dataset):
    syn.generator.eval()
    rootdir = join(
        "../data_root/patches/" + syn.config.data_name + '/' + syn.config.generator + '_' + syn.config.helper)
    if not os.path.isdir(rootdir):
        os.makedirs(rootdir)
    nrow = 8
    with torch.no_grad():
        for path in vali_dataset.patches_imgs:
            base_name = os.path.basename(path)[:-4]
            print(join(rootdir,base_name))
            img = np.load(join(path))
            img_ori = torch.from_numpy(np.expand_dims(img[..., 0], 0)).unsqueeze(0)
            img_erase = torch.from_numpy(np.expand_dims(img[..., 1], 0)).unsqueeze(0)
            tumor_region = torch.from_numpy(np.expand_dims(img[..., 2], 0)).unsqueeze(0)
            boundary_region = torch.from_numpy(np.expand_dims(img[..., 3], 0)).unsqueeze(0)
            seg = torch.from_numpy(np.expand_dims(img[..., 4], 0)).unsqueeze(0)

            batch = {
                "img_erase": img_erase,
                "img_ori": img_ori,
                "tumor_region": tumor_region,
                "boundary_region": boundary_region,
                "img_seg": seg
            }
            batch_gen = syn.generate_batch(batch)
            batch_ret = syn.test_one_batch(batch_gen)
            fake_B = batch_ret['fake_B']
            img[..., 0] = fake_B.detach().cpu().numpy()
            np.save(join(rootdir, os.path.basename(path)), img)
            save_dir = rootdir + '/test_epoch_gener_gen'
            if not isdir(save_dir):
                os.makedirs(save_dir)
            visual_batch(batch_gen['real_A'], save_dir, "%s_real_A" % (base_name), channel=1, nrow=nrow)
            visual_batch(batch_gen['real_B'], save_dir, "%s_real_B" % (base_name), channel=1, nrow=nrow)
            visual_batch(batch_gen['boundary_B'], save_dir, "%s_boundary_B" % (base_name), channel=1,
                         nrow=nrow)
            visual_batch(batch_gen['tumor_B'], save_dir, "%s_tumor_B" % (base_name), channel=1,
                         nrow=nrow)
            visual_batch(torch.clamp(torch.from_numpy(np.expand_dims(img[..., 0], 0)).unsqueeze(0) * 255 + 0.5, 0, 255), save_dir,
                         "%s_gen_B" % (base_name), channel=1,
                         nrow=nrow)
            if batch_ret['fake_B_coarse'] is not None:
                visual_batch(torch.clamp(batch_ret['fake_B_coarse'] * 255 + 0.5, 0, 255), save_dir,
                             "%s_gen_B_coarse" % (base_name),
                             channel=1,
                             nrow=nrow)
            if batch_ret['boundary'] is not None:
                visual_batch(torch.clamp(batch_ret['boundary'][-1] * 255 + 0.5, 0, 255), save_dir,
                             "%s_gen_boundary" % (base_name),
                             channel=1,
                             nrow=nrow)
            if batch_ret['boundary_coarse'] is not None:
                visual_batch(torch.clamp(batch_ret['boundary_coarse'][-1] * 255 + 0.5, 0, 255), save_dir,
                             "%s_gen_boundary_coarse" % (base_name),
                             channel=1,
                             nrow=nrow)
        zipDir(save_dir, save_dir + '.zip')
        shutil.rmtree(save_dir)


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
    argparser.add_argument('--train', help='test not need write', default=False)
    argparser.add_argument('--gpu', help='GPU 0,1,2,3', default=0)
    argparser.add_argument('--gpu-count', help='number of GPUs (0,1,2,3)', default='0')
    argparser.add_argument('--run-num', help='run num: 0,2,3', default=0)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads((config.workers + 1) * len(config.gpu_count))

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
