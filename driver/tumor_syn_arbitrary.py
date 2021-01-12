import matplotlib
import sys

sys.path.extend(["../", "./"])
from utils.utils import *
from os.path import join, isdir
import time
from models import MODELS
import torch
from torch.cuda import empty_cache
import random
from driver.helper import HELPER
from driver.Config import Configurable
from module.Critierion import L1LossMaskedMean, VGGLoss, StyleLoss
from models.vgg import Vgg16
from PIL import Image

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse

DURATION = 20


def main(config):
    # Initialize generator and discriminator
    generator = MODELS[config.generator](2, 1)
    discriminator = MODELS[config.discriminator](nc_in=1)
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

    helper = config.helper
    print("helper:\t" + helper)
    syn = HELPER[helper](generator, discriminator,
                         criterion, config)
    weight_files = sorted(glob(join(syn.config.load_model_path, 'checkpoint_epoch_*.pth')), reverse=True)

    print("loaded:" + weight_files[0])
    syn.load_generator_history_weight(weight_files[0])

    syn.move_to_cuda()
    test(syn)


def get_random_mask(path='/root/qgking/tumor_synthesis/data_root/patches/masks'):
    files = os.listdir(path)
    random_idx = np.random.randint(0, len(files))
    mask = Image.open(join(path, files[random_idx]))
    mask = mask.convert('L')
    mask = np.where(np.array(mask) > 0, 1, 0)
    mask = np.tile(np.expand_dims(mask, -1), 15)
    mm = np.zeros((mask.shape[0], mask.shape[0], mask.shape[0]))
    mm[..., 25:(25 + 15)] = mask
    return mm


def test(syn):
    syn.generator.eval()
    epoch_time = Averagvalue()
    end = time.time()
    scale = 2
    if syn.config.data_name == 'BraTS18':
        from tumor_data.bak.BRATSProcess import get_image_paths
        image_paths = get_image_paths()
        img_idx = 0
        for item in image_paths:
            print(item)
            image = read_image(item)
            t1ce_img = image[..., np.random.randint(0, 4)]
            msk = get_random_mask()
            x_start = t1ce_img.shape[0] // 4
            x_range = t1ce_img.shape[0] // 2
            y_start = t1ce_img.shape[1] // 4
            y_range = t1ce_img.shape[1] // 2
            z_start = t1ce_img.shape[2] // 4
            z_range = t1ce_img.shape[2] // 2
            while True:
                mminx = int(x_start + np.random.randint(0, int(msk.shape[0] / x_range * scale)))
                mmaxx = mminx + msk.shape[0]
                mminy = int(y_start + np.random.randint(0, int(msk.shape[0] / y_range * scale)))
                mmaxy = mminy + msk.shape[0]
                mminz = int(z_start + np.random.randint(0, int(msk.shape[0] / z_range * scale)))
                mmaxz = mminz + msk.shape[0]
                real_B = t1ce_img[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
                real_BB = real_B.copy()
                if real_B.shape != (msk.shape[0], msk.shape[0], msk.shape[0]):
                    continue
                break
            real_A = real_B * (1 - msk) + msk
            fake_B = infer(syn, real_A, msk)
            t1ce_img[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz] = fake_B
            image_input = t1ce_img.copy()
            image_input[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz] = real_A
            vis(img_idx, syn, real_A, real_BB, fake_B, t1ce_img, image_input, msk)
            img_idx += 1
    elif syn.config.data_name == 'LiTS17':
        from tumor_data.bak.LITSProcess import get_image_paths, load_one_case
        image_paths = get_image_paths()
        img_idx = 0
        for item in image_paths:
            print(item)
            image, seg = load_one_case(item)
            msk = get_random_mask()
            time_s = time.time()
            duration = 0
            while True:
                mminx, mmaxx, mminy, mmaxy, mminz, mmaxz = cal_min_max(seg, msk, scale=scale)
                real_B = image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
                real_BB = real_B.copy()
                if real_B.shape != (msk.shape[0], msk.shape[0], msk.shape[0]):
                    time_e = time.time()
                    duration = time_e - time_s
                    if duration > DURATION:
                        break
                    continue
                break
            if duration > DURATION:
                continue
            real_A = real_B * (1 - msk) + msk
            fake_B = infer(syn, real_A, msk)
            image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz] = fake_B
            image_input = image.copy()
            image_input[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz] = real_A
            vis(img_idx, syn, real_A, real_BB, fake_B, image, image_input, msk)
            img_idx += 1
    elif syn.config.data_name == 'KiTS19':
        from tumor_data.bak.KITSProcess import get_image_paths, load_one_case
        path_list = get_image_paths()
        for subsetindex in range(0, 210, 1):
            print(path_list[subsetindex])
            image, seg = load_one_case(path_list, subsetindex)
            msk = get_random_mask()
            seg = np.transpose(seg, (1, 2, 0))
            image = np.transpose(image, (1, 2, 0))
            duration = 0
            time_s = time.time()
            while True:
                mminx, mmaxx, mminy, mmaxy, mminz, mmaxz = cal_min_max(seg, msk, scale=scale)
                real_B = image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
                real_BB = real_B.copy()
                if real_B.shape != (msk.shape[0], msk.shape[0], msk.shape[0]):
                    time_e = time.time()
                    duration = time_e - time_s
                    if duration > DURATION:
                        break
                    continue
                break
            if duration > DURATION:
                continue
            real_A = real_B * (1 - msk) + msk
            fake_B = infer(syn, real_A, msk)
            image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz] = fake_B
            image_input = image.copy()
            image_input[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz] = real_A
            vis(subsetindex, syn, real_A, real_BB, fake_B, image, image_input, msk)
            subsetindex += 1
    elif syn.config.data_name == 'LUNA16':
        from tumor_data.bak.LUNAProcess import ROOT_DIR, PROJ_DIR, make_mask, set_bounds, MIN_IMG_BOUND, MAX_IMG_BOUND, \
            get_filename
        import pandas as pd
        import SimpleITK as sitk

        tqdm = lambda x: x
        seriesindex = 0
        for subsetindex in range(10):
            luna_path = join("/root/data", "LUNA16/data/")
            luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
            output_path = join(ROOT_DIR, PROJ_DIR, "data_root/LUNA/")
            file_list = glob(luna_subset_path + "*.mhd")

            # The locations of the nodes
            df_node = pd.read_csv(luna_path + "annotations.csv")
            df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
            df_node = df_node.dropna()
            # Looping over the image files
            for fcount, img_file in enumerate(tqdm(file_list)):
                # get all nodules associate with file
                mini_df = df_node[df_node["file"] == img_file]
                # load the src data once
                itk_img = sitk.ReadImage(img_file)

                print(img_file)
                # indexes are z,y,x (notice the ordering)
                img_array = sitk.GetArrayFromImage(itk_img)
                # num_z height width constitute the transverse plane
                num_z, height, width = img_array.shape
                # x,y,z  Origin in world coordinates (mm)
                origin = np.array(itk_img.GetOrigin())
                # spacing of voxels in world coor. (mm)
                spacing = np.array(itk_img.GetSpacing())
                # some files may not have a nodule--skipping those
                if mini_df.shape[0] == 0:
                    # set out mask data once
                    mask_itk = np.zeros(shape=(num_z, height, width), dtype=np.float)
                    mask_itk = np.uint8(mask_itk * 255.)
                    mask_itk = np.clip(mask_itk, 0, 255).astype('uint8')
                    sitk_maskimg = sitk.GetImageFromArray(mask_itk)
                    sitk_maskimg.SetSpacing(spacing)
                    sitk_maskimg.SetOrigin(origin)
                    # substring of input file path ang save output mask file
                    sub_img_file = img_file[len(luna_subset_path):-4]
                    # sitk.WriteImage(sitk_maskimg, luna_subset_mask_path + sub_img_file + "_segmentation.mhd")
                if mini_df.shape[0] > 0:
                    # set out mask data once
                    mask_itk = np.zeros(shape=(num_z, height, width), dtype=np.float)
                    # go through all nodes in one series image
                    for node_idx, cur_row in mini_df.iterrows():
                        node_x = cur_row["coordX"]
                        node_y = cur_row["coordY"]
                        node_z = cur_row["coordZ"]
                        diam = cur_row["diameter_mm"]
                        center = np.array([node_x, node_y, node_z])
                        # nodule center
                        v_center = np.rint((center - origin) / spacing)
                        # nodule diam
                        # v_diam = np.rint((diam - origin) / spacing)
                        v_diam = int(diam / spacing[0])
                        # convert x,y,z order v_center to z,y,z order v_center
                        v_center[0], v_center[1], v_center[2] = v_center[2], v_center[1], v_center[0]
                        # make_mask(mask_itk, v_center, v_diam)
                        make_mask(mask_itk, v_center, diam)

                    mask_itk = np.uint8(mask_itk * 255.)
                    mask_itk = np.clip(mask_itk, 0, 255).astype('uint8')
                    sitk_maskimg = sitk.GetImageFromArray(mask_itk)
                    sitk_maskimg.SetSpacing(spacing)
                    sitk_maskimg.SetOrigin(origin)
                    img_array = set_bounds(img_array, MIN_IMG_BOUND, MAX_IMG_BOUND)
                    image = np.transpose(img_array, (1, 2, 0))
                    image = normalize_(image)
                    seg = np.where(mask_itk > 2, 1, 0)
                    seg = np.transpose(seg, (1, 2, 0))

                    seg = label_connected_component(seg)
                    seg_unique = np.unique(seg)[1:]
                    print(seg_unique)
                    for unique in seg_unique:
                        seg_uni = seg.copy()
                        # image_uni = image.copy()
                        total_pixels = np.sum(seg_uni == unique)
                        if total_pixels < 100:
                            print("total pixels less skip")
                            continue
                        print("total pixels:" + str(total_pixels))
                        # image_uni[np.where(seg_uni == unique)] = 1
                        seg_uni = np.where(seg_uni != unique, 0, 1)
                        minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=0, inferior=0)
                        print((maxx - minx, maxy - miny, maxz - minz))

                        msk = get_random_mask()
                        duration = 0
                        time_s = time.time()
                        while True:
                            mminx, mmaxx, mminy, mmaxy, mminz, mmaxz = cal_min_max_luna(seg_uni, msk, scale=scale)
                            real_B = image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
                            real_BB = real_B.copy()
                            if real_B.shape != (msk.shape[0], msk.shape[0], msk.shape[0]):
                                time_e = time.time()
                                duration = time_e - time_s
                                if duration > DURATION:
                                    break
                                continue
                            break
                        if duration > DURATION:
                            continue

                        # real_A = real_B * (1 - seg_uni[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]) + seg_uni[mminx:mmaxx,
                        #                                                                          mminy:mmaxy,
                        #                                                                          mminz:mmaxz]
                        real_A = real_B * (1 - msk) + msk
                        # save_dir = syn.config.tmp_dir + '/arbi/test_img_tttt'
                        # if not isdir(save_dir):
                        #     os.makedirs(save_dir)
                        # for sss in range(real_B.shape[-1]):
                        #     visualize(np.expand_dims(real_B[..., sss], -1), join(save_dir, str(sss) + "Agen"))
                        #     visualize(np.expand_dims(seg_uni[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz][..., sss], -1),
                        #               join(save_dir, str(sss) + "Binput"))
                        #     visualize(np.expand_dims(real_A[..., sss], -1),
                        #               join(save_dir, str(sss) + "Cinput"))
                        # msk = seg_uni[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz].copy()
                        fake_B = infer(syn, real_A, msk)
                        image[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz] = fake_B
                        image_input = image.copy()
                        image_input[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz] = real_A
                        vis(seriesindex, syn, real_A, real_BB, fake_B, image, image_input, msk)
                        seriesindex += 1
    else:
        raise ValueError("not support data!")

    # del loss, prec1
    empty_cache()
    # measure elapsed time
    epoch_time.update(time.time() - end)
    info = 'Test Time {batch_time.avg:.3f} '.format(batch_time=epoch_time)
    print(info)


def cal_min_max(raw_img, msk, scale=2):
    # minx, maxx, miny, maxy, minz, maxz = min_max_voi(raw_img, superior=0, inferior=0)
    # rand_loc = [-3, -2.5 - 2, -1.5, -1, 1, 1.5, 2, 2.5, 3]
    # rand_idx = np.random.randint(0, len(rand_loc))
    # mminx = int(minx + (maxx - minx) * rand_loc[rand_idx])
    # mmaxx = mminx + msk.shape[0]
    # rand_idx = np.random.randint(0, len(rand_loc))
    # mminy = int(miny + (maxy - miny) * rand_loc[rand_idx])
    # mmaxy = mminy + msk.shape[0]
    # rand_idx = np.random.randint(0, len(rand_loc))
    # mminz = int(minz + (maxz - minz) * rand_loc[rand_idx])
    # mmaxz = mminz + msk.shape[0]
    old_idx = np.where(raw_img > 0)
    mminx = np.random.randint(0, len(old_idx[0]))
    mminx = old_idx[0][mminx]
    mmaxx = mminx + msk.shape[0]
    mminy = np.random.randint(0, len(old_idx[1]))
    mminy = old_idx[1][mminy]
    mmaxy = mminy + msk.shape[0]
    mminz = np.random.randint(0, len(old_idx[2]))
    mminz = old_idx[2][mminz]
    mmaxz = mminz + msk.shape[0]
    return mminx, mmaxx, mminy, mmaxy, mminz, mmaxz


def cal_min_max_luna(raw_img, msk, scale=2):
    minx, maxx, miny, maxy, minz, maxz = min_max_voi(raw_img, superior=0, inferior=0)
    rand_loc = [ -2.5 - 2, -1.5, -0.5, 0, 0.5, -1, 1, 1.5, 2, 2.5]
    rand_idx = np.random.randint(0, len(rand_loc))
    mminx = int(minx + (maxx - minx) * rand_loc[rand_idx])
    mmaxx = mminx + msk.shape[0]
    rand_idx = np.random.randint(0, len(rand_loc))
    mminy = int(miny + (maxy - miny) * rand_loc[rand_idx])
    mmaxy = mminy + msk.shape[0]
    rand_idx = np.random.randint(0, len(rand_loc))
    mminz = int(minz + (maxz - minz) * rand_loc[rand_idx])
    mmaxz = mminz + msk.shape[0]
    return mminx, mmaxx, mminy, mmaxy, mminz, mmaxz


def infer(syn, real_A, msk):
    with torch.no_grad():
        FloatTensor = torch.cuda.FloatTensor if syn.config.use_cuda else torch.FloatTensor
        realA = np.expand_dims(real_A, 0)
        realA = np.expand_dims(realA, 0)
        realA = torch.from_numpy(realA)
        tumorB = np.expand_dims(msk, 0)
        tumorB = np.expand_dims(tumorB, 0)
        tumorB = torch.from_numpy(tumorB)
        if syn.config.use_cuda:
            realA = realA.cuda(syn.config.gpu).float()
            tumorB = tumorB.cuda(syn.config.gpu).float()
        realA = FloatTensor(realA).requires_grad_(False)
        tumorB = FloatTensor(tumorB).requires_grad_(False)
        g_out = syn.generator(realA, tumorB)
        fake_B = g_out['outputs']
        fake_B = np.squeeze(fake_B.detach().cpu().numpy(), 0)
        fake_B = np.squeeze(fake_B, 0)
    return fake_B


def vis(img_idx, syn, real_A, real_B, fake_B, image, image_in, msk):
    save_dir = syn.config.tmp_dir + '/arbi/test_img_' + str(img_idx)
    if not isdir(save_dir):
        os.makedirs(save_dir)
    # viz_ims = overlay(fake_B, msk, msk, alpha=0.3)
    _, _, _, _, minz, maxz = min_max_voi(msk, superior=0, inferior=0)
    for sss in range(minz, maxz):
        visualize(np.expand_dims(fake_B[..., sss], -1), join(save_dir, str(sss) + "Agen"))
        visualize(np.expand_dims(real_A[..., sss], -1), join(save_dir, str(sss) + "Binput"))
        visualize(np.expand_dims(real_B[..., sss], -1), join(save_dir, str(sss) + "Creal"))
        # fpath = join(save_dir, str(sss) + "Doverlay.png")
        # scipy.misc.imsave(str(fpath), viz_ims[:, :, sss, :])
    # save_dir = syn.config.tmp_dir + '/arbi/test_input_img_' + str(img_idx)
    # if not isdir(save_dir):
    #     os.makedirs(save_dir)
    # for sss in range(image.shape[-1]):
    #     visualize(np.expand_dims(image[..., sss], -1), join(save_dir, str(sss) + "A_all_fake"))
    #     visualize(np.expand_dims(image_in[..., sss], -1), join(save_dir, str(sss) + "B_all_input"))


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

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads(args.thread)
    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
