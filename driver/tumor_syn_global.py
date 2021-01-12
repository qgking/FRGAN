import matplotlib
import sys

sys.path.extend(["../", "./"])
from utils.utils import *

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from os.path import join
from models import MODELS
import torch
import random
from driver.helper import HELPER
from driver.Config import Configurable
from module.Critierion import L1LossMaskedMean, L2LossMaskedMean

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse

DURATION = 20


def main(config):
    # Initialize generator and discriminator
    generator = MODELS[config.generator](2, 1)
    discriminator = MODELS[config.discriminator](nc_in=1)
    criterion = {
        'criterion_GAN': torch.nn.L1Loss(),
        'criterion_pixelwise': torch.nn.L1Loss(),
        'L1LossMaskedMean': L1LossMaskedMean(),
        "L2LossMaskedMean": L2LossMaskedMean()
    }

    helper = config.helper
    print("helper:\t" + helper)
    syn = HELPER[helper](generator, discriminator,
                         criterion, config)
    weight_files = sorted(glob(join(syn.config.load_model_path, 'checkpoint_epoch_*.pth')), reverse=True)

    print("loaded:" + weight_files[0])
    syn.load_generator_history_weight(weight_files[0])
    syn.move_to_cuda()
    syn.generator.eval()
    dirdir = join(config.tmp_dir, 'magviewtest')
    if not os.path.isdir(dirdir):
        os.makedirs(dirdir)
    if syn.config.data_name == 'BraTS18':
        from tumor_data.bak.BRATSProcess import get_image_paths
        image_paths = get_image_paths()
        img_idx = 0
        image_paths.reverse()
        for item in image_paths:
            print(item)
            img_idx += 1
            if img_idx > 50:
                break
            image = read_image(item)
            t1ce_img = image[..., np.random.randint(0, 4)]
            seg = image[..., -1]
            seg = label_connected_component(seg)
            seg_unique = np.unique(seg)[1:]
            print(seg_unique)
            for unique in seg_unique:
                seg_uni = seg.copy()
                total_pixels = np.sum(seg_uni == unique)
                if total_pixels < 100:
                    print("total pixels less skip")
                    continue
                print("total pixels:" + str(total_pixels))
                print(unique)
                # image_uni[np.where(seg_uni == unique)] = 1
                seg_uni = np.where(seg_uni != unique, 0, 1)
                minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=10, inferior=10)
                _, _, _, _, bottom, top = min_max_voi(seg_uni, superior=0, inferior=0)

                print((maxx - minx, maxy - miny, maxz - minz))
                mmaxz = maxz - (maxz - minz) % 4
                mmaxy = maxy - (maxy - miny) % 4
                mmaxx = maxx - (maxx - minx) % 4
                print((minx, mmaxx, miny, mmaxy, minz, mmaxz))
                ori_npy = t1ce_img[minx:mmaxx, miny:mmaxy, minz:mmaxz]
                seg_npy = seg_uni[minx:mmaxx, miny:mmaxy, minz:mmaxz]

                seg_npy = np.where(seg_npy > 0, 1, 0)
                tumor_npy = seg_npy.copy()

                msk_npy = ori_npy.copy()
                msk_npy[seg_npy > 0] = 1

                image_patch = torch.from_numpy(np.expand_dims(np.expand_dims(msk_npy, 0), 0))
                tumor_region = torch.from_numpy(np.expand_dims(np.expand_dims(tumor_npy, 0), 0))
                if syn.use_cuda:
                    image_patch = image_patch.cuda(syn.device).float()
                    tumor_region = tumor_region.cuda(syn.device).float()
                real_A = syn.FloatTensor(image_patch).requires_grad_(False)
                tumor_B = syn.FloatTensor(tumor_region).requires_grad_(False)
                with torch.no_grad():
                    g_out = syn.generator(real_A, tumor_B)
                fake_B = g_out['outputs']
                fake_B = torch.squeeze(torch.squeeze(fake_B, 0), 0)
                final_img = t1ce_img.copy()
                final_img[minx:mmaxx, miny:mmaxy, minz:mmaxz] = fake_B.detach().cpu().numpy()
                for sss in range(bottom, top):
                    visualize(np.expand_dims(seg_uni[:, :, sss], -1),
                              join(dirdir, 'img_' + '{0:0>3}'.format(img_idx) + '_' + '{0:0>1}'.format(
                                  unique) + '_' + str(sss) + "_Aseg"))
                    visualize(np.expand_dims(final_img[:, :, sss], -1),
                              join(dirdir, 'img_' + '{0:0>3}'.format(img_idx) + '_' + '{0:0>1}'.format(
                                  unique) + '_' + str(sss) + "_Bsyn"))
                    miccaiimshow(final_img[:, :, sss], seg_uni[:, :, sss],
                                 join(dirdir, 'img_' + '{0:0>3}'.format(img_idx) + '_' + '{0:0>1}'.format(
                                     unique) + '_' + str(sss) + "_Cmag"))

    elif syn.config.data_name == 'LiTS17':
        from tumor_data.bak.LITSProcess import get_image_paths, load_one_case
        image_paths = get_image_paths()
        image_paths = sorted(image_paths, reverse=True)
        items_idx = 0
        for item in image_paths:
            items_idx += 1
            if items_idx == 15:
                break
            print(item)
            image, seg = load_one_case(item)
            seg = np.where(seg > 1, 1, 0)
            seg = label_connected_component(seg)
            seg_unique = np.unique(seg)[1:]
            print(seg_unique)
            for unique in seg_unique:
                seg_uni = seg.copy()
                total_pixels = np.sum(seg_uni == unique)
                if total_pixels < 400:
                    print("total pixels less skip")
                    continue
                print("total pixels:" + str(total_pixels))
                print(unique)
                # image_uni[np.where(seg_uni == unique)] = 1
                seg_uni = np.where(seg_uni != unique, 0, 1)
                minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=10, inferior=10)
                _, _, _, _, bottom, top = min_max_voi(seg_uni, superior=0, inferior=0)

                print((maxx - minx, maxy - miny, maxz - minz))
                mmaxz = maxz - (maxz - minz) % 4
                mmaxy = maxy - (maxy - miny) % 4
                mmaxx = maxx - (maxx - minx) % 4
                print((minx, mmaxx, miny, mmaxy, minz, mmaxz))
                ori_npy = image[minx:mmaxx, miny:mmaxy, minz:mmaxz]
                seg_npy = seg_uni[minx:mmaxx, miny:mmaxy, minz:mmaxz]

                seg_npy = np.where(seg_npy > 0, 1, 0)
                tumor_npy = seg_npy.copy()

                msk_npy = ori_npy.copy()
                msk_npy[seg_npy > 0] = 1

                image_patch = torch.from_numpy(np.expand_dims(np.expand_dims(msk_npy, 0), 0))
                tumor_region = torch.from_numpy(np.expand_dims(np.expand_dims(tumor_npy, 0), 0))
                if syn.use_cuda:
                    image_patch = image_patch.cuda(syn.device).float()
                    tumor_region = tumor_region.cuda(syn.device).float()
                real_A = syn.FloatTensor(image_patch).requires_grad_(False)
                tumor_B = syn.FloatTensor(tumor_region).requires_grad_(False)
                with torch.no_grad():
                    g_out = syn.generator(real_A, tumor_B)
                fake_B = g_out['outputs']
                fake_B = torch.squeeze(torch.squeeze(fake_B, 0), 0)
                final_img = image.copy()
                final_img[minx:mmaxx, miny:mmaxy, minz:mmaxz] = fake_B.detach().cpu().numpy()
                for sss in range(bottom, top):
                    visualize(np.expand_dims(seg_uni[:, :, sss], -1),
                              join(dirdir, 'img_' + '{0:0>3}'.format(items_idx) + '_' + '{0:0>1}'.format(
                                  unique) + '_' + str(sss) + "_Aseg"))
                    visualize(np.expand_dims(final_img[:, :, sss], -1),
                              join(dirdir, 'img_' + '{0:0>3}'.format(items_idx) + '_' + '{0:0>1}'.format(
                                  unique) + '_' + str(sss) + "_Bsyn"))
                    miccaiimshow(final_img[:, :, sss], seg_uni[:, :, sss],
                                 join(dirdir, 'img_' + '{0:0>3}'.format(items_idx) + '_' + '{0:0>1}'.format(
                                     unique) + '_' + str(sss) + "_Cmag"))
    elif syn.config.data_name == 'KiTS19':
        from tumor_data.bak.KITSProcess import get_image_paths, load_one_case
        path_list = get_image_paths()
        for subsetindex in range(209, 180, -1):
            print(path_list[subsetindex])
            image, seg = load_one_case(path_list, subsetindex)
            seg = np.where(seg > 1, 1, 0)
            seg = label_connected_component(seg)
            print(seg.shape)
            seg_unique = np.unique(seg)[1:]
            seg = np.transpose(seg, (1, 2, 0))
            image = np.transpose(image, (1, 2, 0))
            print(seg_unique)
            for unique in seg_unique:
                seg_uni = seg.copy()
                total_pixels = np.sum(seg_uni == unique)
                if total_pixels < 400:
                    print("total pixels less skip")
                    continue
                print("total pixels:" + str(total_pixels))
                print(unique)
                # image_uni[np.where(seg_uni == unique)] = 1
                seg_uni = np.where(seg_uni != unique, 0, 1)
                minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=10, inferior=10)
                _, _, _, _, bottom, top = min_max_voi(seg_uni, superior=0, inferior=0)

                print((maxx - minx, maxy - miny, maxz - minz))
                mmaxz = maxz - (maxz - minz) % 4
                mmaxy = maxy - (maxy - miny) % 4
                mmaxx = maxx - (maxx - minx) % 4
                print((minx, mmaxx, miny, mmaxy, minz, mmaxz))
                ori_npy = image[minx:mmaxx, miny:mmaxy, minz:mmaxz]
                seg_npy = seg_uni[minx:mmaxx, miny:mmaxy, minz:mmaxz]

                seg_npy = np.where(seg_npy > 0, 1, 0)
                tumor_npy = seg_npy.copy()

                msk_npy = ori_npy.copy()
                msk_npy[seg_npy > 0] = 1

                image_patch = torch.from_numpy(np.expand_dims(np.expand_dims(msk_npy, 0), 0))
                tumor_region = torch.from_numpy(np.expand_dims(np.expand_dims(tumor_npy, 0), 0))
                if syn.use_cuda:
                    image_patch = image_patch.cuda(syn.device).float()
                    tumor_region = tumor_region.cuda(syn.device).float()
                real_A = syn.FloatTensor(image_patch).requires_grad_(False)
                tumor_B = syn.FloatTensor(tumor_region).requires_grad_(False)
                with torch.no_grad():
                    g_out = syn.generator(real_A, tumor_B)
                fake_B = g_out['outputs']
                fake_B = torch.squeeze(torch.squeeze(fake_B, 0), 0)
                final_img = image.copy()
                final_img[minx:mmaxx, miny:mmaxy, minz:mmaxz] = fake_B.detach().cpu().numpy()
                for sss in range(bottom, top):
                    visualize(np.expand_dims(seg_uni[:, :, sss], -1),
                              join(dirdir, 'img_' + '{0:0>3}'.format(subsetindex) + '_' + '{0:0>1}'.format(
                                  unique) + '_' + str(sss) + "_Aseg"))
                    visualize(np.expand_dims(final_img[:, :, sss], -1),
                              join(dirdir, 'img_' + '{0:0>3}'.format(subsetindex) + '_' + '{0:0>1}'.format(
                                  unique) + '_' + str(sss) + "_Bsyn"))
                    miccaiimshow(final_img[:, :, sss], seg_uni[:, :, sss],
                                 join(dirdir, 'img_' + '{0:0>3}'.format(subsetindex) + '_' + '{0:0>1}'.format(
                                     unique) + '_' + str(sss) + "_Cmag"))
    elif syn.config.data_name == 'LUNA16':
        from tumor_data.bak.LUNAProcess import ROOT_DIR, PROJ_DIR, make_mask, set_bounds, MIN_IMG_BOUND, MAX_IMG_BOUND, \
            get_filename
        import pandas as pd
        import SimpleITK as sitk
        tqdm = lambda x: x
        countcount = 0
        for subsetindex in range(9, 4, -1):
            luna_path = join("/root/data", "LUNA16/data/")
            luna_subset_path = luna_path + "subset" + str(subsetindex) + "/"
            output_path = join(ROOT_DIR, PROJ_DIR, "data_root/LUNA/")
            file_list = sorted(glob(luna_subset_path + "*.mhd"), reverse=True)

            # The locations of the nodes
            df_node = pd.read_csv(luna_path + "annotations.csv")
            df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
            df_node = df_node.dropna()
            # Looping over the image files
            for fcount, img_file in enumerate(tqdm(file_list)):
                countcount += 1
                if countcount > 70:
                    break
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
                        print(unique)
                        # image_uni[np.where(seg_uni == unique)] = 1
                        seg_uni = np.where(seg_uni != unique, 0, 1)
                        minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=10, inferior=10)
                        _, _, _, _, bottom, top = min_max_voi(seg_uni, superior=0, inferior=0)

                        print((maxx - minx, maxy - miny, maxz - minz))
                        mmaxz = maxz - (maxz - minz) % 4
                        mmaxy = maxy - (maxy - miny) % 4
                        mmaxx = maxx - (maxx - minx) % 4
                        print((minx, mmaxx, miny, mmaxy, minz, mmaxz))
                        ori_npy = image[minx:mmaxx, miny:mmaxy, minz:mmaxz]
                        seg_npy = seg_uni[minx:mmaxx, miny:mmaxy, minz:mmaxz]

                        seg_npy = np.where(seg_npy > 0, 1, 0)
                        tumor_npy = seg_npy.copy()

                        msk_npy = ori_npy.copy()
                        msk_npy[seg_npy > 0] = 1

                        image_patch = torch.from_numpy(np.expand_dims(np.expand_dims(msk_npy, 0), 0))
                        tumor_region = torch.from_numpy(np.expand_dims(np.expand_dims(tumor_npy, 0), 0))
                        if syn.use_cuda:
                            image_patch = image_patch.cuda(syn.device).float()
                            tumor_region = tumor_region.cuda(syn.device).float()
                        real_A = syn.FloatTensor(image_patch).requires_grad_(False)
                        tumor_B = syn.FloatTensor(tumor_region).requires_grad_(False)
                        with torch.no_grad():
                            g_out = syn.generator(real_A, tumor_B)
                        fake_B = g_out['outputs']
                        fake_B = torch.squeeze(torch.squeeze(fake_B, 0), 0)
                        final_img = image.copy()
                        final_img[minx:mmaxx, miny:mmaxy, minz:mmaxz] = fake_B.detach().cpu().numpy()
                        for sss in range(bottom, top):
                            visualize(np.expand_dims(seg_uni[:, :, sss], -1),
                                      join(dirdir, 'img_' + '{0:0>3}'.format(subsetindex) + '_' + '{0:0>1}'.format(
                                          unique) + '_' + str(sss) + "_Aseg"))
                            visualize(np.expand_dims(final_img[:, :, sss], -1),
                                      join(dirdir, 'img_' + '{0:0>3}'.format(subsetindex) + '_' + '{0:0>1}'.format(
                                          unique) + '_' + str(sss) + "_Bsyn"))
                            miccaiimshow(final_img[:, :, sss], seg_uni[:, :, sss],
                                         join(dirdir, 'img_' + '{0:0>3}'.format(subsetindex) + '_' + '{0:0>1}'.format(
                                             unique) + '_' + str(sss) + "_Cmag"))
            if countcount > 70:
                break

    else:
        raise ValueError("not support data!")


def miccaiimshow(img, seg, fname, titles=None, plot_separate_img=True):
    """Takes raw image img, seg in range 0-2, list of predictions in range 0-2"""
    plt.figure(figsize=(25, 12))
    ALPHA = 1
    boundary_npy = ((
                        filters.gaussian_filter(
                            255. * np.multiply(np.invert(morph.binary_erosion(seg)), seg),
                            1.0)) / 255).reshape(seg.shape)
    boundary_npy = np.where(boundary_npy > np.max(boundary_npy) / 2, 1, 0)
    preds = boundary_npy
    if len(preds.shape) == 3:
        n_plots = len(preds)
    else:
        n_plots = 1
    subplot_offset = 0

    plt.set_cmap('gray')

    if plot_separate_img:
        n_plots += 1
        subplot_offset = 1
        plt.subplot(1, n_plots, 1)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.title("Image")
        plt.axis('off')
        plt.imshow(img, cmap="gray")
    if type(preds) != list:
        preds = [preds]
    for i, pred in enumerate(preds):
        liver = pred == 1
        diffliver = np.logical_xor(seg == 1, liver)
        plt.subplot(1, n_plots, i + 1 + subplot_offset)
        title = titles[i] if titles is not None and i < len(titles) else ""
        plt.title(title)
        plt.axis('off')
        plt.imshow(img);

        # Liver prediction
        plt.imshow(np.ma.masked_where(liver == 0, liver), cmap="Greens", vmin=0.1, vmax=1.2, alpha=ALPHA)
        # Liver : Pixels in ground truth, not in prediction
        # plt.imshow(np.ma.masked_where(diffliver == 0, diffliver), cmap="Spectral", vmin=0.1, vmax=2.2, alpha=ALPHA)

    plt.savefig(fname, transparent=True)
    plt.close()


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
    argparser.add_argument('--run-num', help='run num: 0,2,3', default=0)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads((config.workers + 1) * len(config.gpu_count))

    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)
    main(config)
