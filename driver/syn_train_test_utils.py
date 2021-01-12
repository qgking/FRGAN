# -*- coding: utf-8 -*-
# @Time    : 20/9/22 16:49
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : syn_train_test.py
import matplotlib
import sys

sys.path.extend(["../", "./"])
from utils.utils import *
from os.path import join, isdir
import time
import torch
from torch.cuda import empty_cache
import shutil
from skimage import morphology
import torch.nn.functional as F
from module.Critierion import L1LossMaskedMean, L2LossMaskedMean

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()

log_template = "[Epoch %d/%d] [Batch %d/%d] [print_time: %.3f] " \
               "[D: %f, G: %f, Content: %f, Content_coarse: %f, " \
               "GAN: %f, " \
               "Tumor: %f, Tumor_Coarse: %f, Boundary: %f, Boundary_Coarse: %f, " \
               "Percep: %f, Style: %f]"


def train(syn, train_loader, model_optimizer, epoch):
    syn.generator.train()
    end = time.time()
    pend = time.time()
    batch_time = Averagvalue()
    printfreq_time = Averagvalue()
    losses_G = Averagvalue()
    losses_D = Averagvalue()
    losses_GAN = Averagvalue()
    losses_pixel = Averagvalue()
    losses_pixel_coarse = Averagvalue()
    losses_tumor = Averagvalue()
    losses_tumor_coarse = Averagvalue()
    losses_boundary = Averagvalue()
    losses_boundary_coarse = Averagvalue()
    losses_style = Averagvalue()
    losses_percp = Averagvalue()
    # ttime = time.time()
    for i, batch in enumerate(train_loader):
        batch_gen = syn.generate_batch(batch)
        # visual_batch(batch_gen['real_A'], syn.config.tmp_dir, "%s_real_A" % (i), channel=1, nrow=8)
        # visual_batch(batch_gen['real_B'], syn.config.tmp_dir, "%s_real_B" % (i), channel=1, nrow=8)
        # visual_batch(batch_gen['boundary_B'], syn.config.tmp_dir, "%s_boundary_B" % (i), channel=1,
        #              nrow=8)
        # visual_batch(batch_gen['tumor_B'], syn.config.tmp_dir, "%s_tumor_B" % (i), channel=1,
        #              nrow=8)
        # ------------------
        #  Train Generators
        # ------------------
        model_optimizer['G'].zero_grad()
        if 'allGateDilateRicherContextRefine' == syn.config.generator:
            batch_ret = syn.train_generator_one_batch_pretrained(batch_gen)
        else:
            batch_ret = syn.train_generator_one_batch(batch_gen)
        loss_GAN = batch_ret['loss_GAN']
        loss_pixel = batch_ret['loss_pixel']
        loss_pixel_coarse = batch_ret['loss_pixel_coarse']
        loss_boundary = batch_ret['loss_boundary']
        loss_boundary_coarse = batch_ret['loss_boundary_coarse']
        loss_tumor = batch_ret['loss_tumor']
        loss_tumor_coarse = batch_ret['loss_tumor_coarse']
        loss_G = batch_ret['loss_G']
        fake_B = batch_ret['fake_B']
        batch_gen['fake_B'] = fake_B
        loss_style = batch_ret['loss_style']
        loss_percep = batch_ret['loss_percep']
        # print("train_generator_one_batch:%s" % (time.time() - ttime))
        # ttime = time.time()
        # model_optimizer['G'].zero_grad()
        model_optimizer['G'].step()

        losses_GAN.update(loss_GAN)
        losses_pixel.update(loss_pixel)
        losses_pixel_coarse.update(loss_pixel_coarse)
        losses_boundary.update(loss_boundary)
        losses_boundary_coarse.update(loss_boundary_coarse)
        losses_tumor.update(loss_tumor)
        losses_tumor_coarse.update(loss_tumor_coarse)
        losses_G.update(loss_G)
        losses_style.update(loss_style)
        losses_percp.update(loss_percep)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        model_optimizer['D'].zero_grad()
        loss_D = syn.train_discriminator_one_batch(batch_gen)

        losses_D.update(loss_D)
        model_optimizer['D'].step()
        # --------------
        #  Log Progress
        # --------------
        batch_time.update(time.time() - end)
        end = time.time()
        # Print log
        if (i + 1) % (int(len(train_loader) / syn.config.printfreq)) == 0:
            printfreq_time.update(time.time() - pend)
            pend = time.time()
            print(
                log_template % (
                    epoch,
                    syn.config.epochs,
                    i,
                    len(train_loader),
                    printfreq_time.val,
                    losses_D.avg,
                    losses_G.avg,
                    losses_pixel.avg,
                    losses_pixel_coarse.avg,
                    losses_GAN.avg,
                    losses_tumor.avg,
                    losses_tumor_coarse.avg,
                    losses_boundary.avg,
                    losses_boundary_coarse.avg,
                    losses_percp.avg,
                    losses_style.avg
                )
            )
            syn.log.flush()
        # print("start loading data%s" % (time.time() - ttime))
        # ttime = time.time()
    empty_cache()
    return {'train/loss_D': losses_D.avg,
            'train/loss_G': losses_G.avg,
            'train/loss_GAN': losses_GAN.avg,
            'train/loss_pixel': losses_pixel.avg,
            'train/loss_tumor': losses_tumor.avg,
            'train/loss_boundary': losses_boundary.avg,
            'train/loss_style': losses_style.avg,
            'train/loss_percep': losses_percp.avg}


def valid(syn, test_loader, epoch):
    syn.generator.eval()
    epoch_time = Averagvalue()
    end = time.time()
    losses_percp = Averagvalue()
    losses_style = Averagvalue()
    losses_pixel = Averagvalue()
    losses_pixel_coarse = Averagvalue()
    losses_tumor = Averagvalue()
    losses_tumor_coarse = Averagvalue()
    losses_boundary = Averagvalue()
    losses_boundary_coarse = Averagvalue()
    nrow = 16
    stage = 'vali'
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch_gen = syn.generate_batch(batch)

            batch_ret = syn.test_one_batch(batch_gen)
            loss_pixel = batch_ret["loss_pixel"]
            loss_pixel_coarse = batch_ret['loss_pixel_coarse']
            loss_boundary = batch_ret['loss_boundary']
            loss_boundary_coarse = batch_ret['loss_boundary_coarse']
            loss_tumor = batch_ret['loss_tumor']
            loss_tumor_coarse = batch_ret['loss_tumor_coarse']
            loss_style = batch_ret['loss_style']
            loss_percep = batch_ret['loss_percep']
            fake_B = batch_ret['fake_B']
            fake_B_coarse = batch_ret['fake_B_coarse']
            boundary = batch_ret['boundary']
            boundary_coarse = batch_ret['boundary_coarse']

            losses_pixel.update(loss_pixel)
            losses_pixel_coarse.update(loss_pixel_coarse)
            losses_boundary.update(loss_boundary)
            losses_boundary_coarse.update(loss_boundary_coarse)
            losses_tumor.update(loss_tumor)
            losses_tumor_coarse.update(loss_tumor_coarse)

            losses_style.update(loss_style)
            losses_percp.update(loss_percep)

            if epoch > syn.config.epochs * 0.7:
                save_dir = syn.config.tmp_dir + '/valid_epoch_' + str(epoch)
                if not isdir(save_dir):
                    os.makedirs(save_dir)
                visual_batch(batch_gen['real_A'], save_dir, "%s_real_A" % (i), channel=1, nrow=nrow)
                visual_batch(batch_gen['real_B'], save_dir, "%s_real_B" % (i), channel=1, nrow=nrow)
                visual_batch(batch_gen['boundary_B'], save_dir, "%s_boundary_B" % (i), channel=1,
                             nrow=nrow)
                visual_batch(batch_gen['tumor_B'], save_dir, "%s_tumor_B" % (i), channel=1,
                             nrow=nrow)
                visual_batch(torch.clamp(fake_B * 255 + 0.5, 0, 255), save_dir, "%s_gen_B" % (i), channel=1,
                             nrow=nrow)
                if fake_B_coarse is not None:
                    visual_batch(torch.clamp(fake_B_coarse * 255 + 0.5, 0, 255), save_dir, "%s_gen_B_coarse" % (i),
                                 channel=1,
                                 nrow=nrow)
                if boundary is not None:
                    visual_batch(torch.clamp(boundary[-1] * 255 + 0.5, 0, 255), save_dir, "%s_gen_boundary" % (i),
                                 channel=1,
                                 nrow=nrow)
                if boundary_coarse is not None:
                    visual_batch(torch.clamp(boundary_coarse[-1] * 255 + 0.5, 0, 255), save_dir,
                                 "%s_gen_boundary_coarse" % (i),
                                 channel=1,
                                 nrow=nrow)

        if epoch > syn.config.epochs * 0.7:
            zipDir(save_dir, save_dir + '.zip')
            shutil.rmtree(save_dir)

        # del loss, prec1
        empty_cache()
        # measure elapsed time
    epoch_time.update(time.time() - end)
    info = 'Valid Epoch Time {batch_time.avg:.3f}, '.format(batch_time=epoch_time) + \
           'Content {loss.avg:f}, '.format(loss=losses_pixel) + \
           'Content Coarse {loss.avg:f}, '.format(loss=losses_pixel_coarse) + \
           'Tumor {loss.avg:f}, '.format(loss=losses_tumor) + \
           'Tumor Coarse {loss.avg:f}, '.format(loss=losses_tumor_coarse) + \
           'Bounrary {loss.avg:f}, '.format(loss=losses_boundary) + \
           'Bounrary Coarse {loss.avg:f}, '.format(loss=losses_boundary_coarse) + \
           'Percp {loss.avg:f}, '.format(loss=losses_percp) + \
           'Style {loss.avg:f}, '.format(loss=losses_style)
    print(info)
    syn.log.flush()
    return {
        stage + '/content_loss': losses_pixel.avg,
        stage + '/tumor_loss': losses_tumor.avg,
        stage + '/boundary_loss': losses_boundary.avg,
        stage + '/percp_loss': losses_percp.avg,
        stage + '/style_loss': losses_style.avg,
    }


def test(syn, test_dataset, epoch):
    syn.generator.eval()
    epoch_time = Averagvalue()
    end = time.time()
    ssims = Averagvalue()
    psnrs = Averagvalue()
    mses = Averagvalue()
    fids = Averagvalue()
    idx = 0
    nrow = 8
    with torch.no_grad():
        for path in test_dataset.patches_imgs:
            base_name = os.path.basename(path)[:-4]
            img = np.load(join(path))
            img_seg = img[..., 1]
            img_ori = img[..., 0]
            seg = label_connected_component(img_seg)
            seg_unique = np.unique(seg)[1:]
            for unique in seg_unique:
                seg_uni = seg.copy()
                img_uni = img_ori.copy()
                # image_uni = image.copy()
                total_pixels = np.sum(seg_uni == unique)
                if 'KiTS' in syn.config.data_name:
                    thres = 100
                    padding = 20
                    patch_z = 64
                if 'LiTS' in syn.config.data_name:
                    thres = 200
                    padding = 20
                    patch_z = 64
                if 'LUNA' in syn.config.data_name:
                    thres = 100
                    padding = 32
                    patch_z = 64
                if total_pixels < thres:
                    # print("total pixels less skip")
                    continue
                # print("total pixels:" + str(total_pixels))
                seg_uni = np.where(seg_uni != unique, 0, 1)
                minx, maxx, miny, maxy, minz, maxz = min_max_voi(seg_uni, superior=0, inferior=0)

                mminx = minx - padding
                mminx = mminx if mminx > 0 else 0
                mmaxx = maxx + padding
                mmaxx = mmaxx if mmaxx < img_uni.shape[0] else img_uni.shape[0] - 1

                mminy = miny - padding
                mminy = mminy if mminy > 0 else 0
                mmaxy = maxy + padding
                mmaxy = mmaxy if mmaxy < img_uni.shape[1] else img_uni.shape[1] - 1

                mminz = minz - padding
                mminz = mminz if mminz > 0 else 0
                mmaxz = maxz + padding
                mmaxz = mmaxz if mmaxz < img_uni.shape[2] else img_uni.shape[2] - 1

                # print((mminx, mmaxx, mminy, mmaxy, mminz, mmaxz))

                ori_npy = img_uni[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
                seg_npy = seg_uni[mminx:mmaxx, mminy:mmaxy, mminz:mmaxz]
                seg_npy[seg_npy > 0] = 1
                aug_patch_img = ori_npy
                aug_patch_seg = seg_npy

                aug_patch_seg = resize(aug_patch_seg, (syn.config.patch_x, syn.config.patch_y,
                                                       syn.config.patch_z), order=0, mode='edge',
                                       cval=0, clip=True, preserve_range=True, anti_aliasing=False)
                aug_patch_img = resize(aug_patch_img, (syn.config.patch_x, syn.config.patch_y,
                                                       syn.config.patch_z), order=3, mode='constant',
                                       cval=0, clip=True, preserve_range=True, anti_aliasing=False)

                id_max = np.max(aug_patch_seg)
                nuclei_inside = np.zeros((syn.config.patch_x, syn.config.patch_y,
                                          patch_z), dtype=np.bool)
                nucleus = aug_patch_seg == id_max
                nuclei_inside += morphology.erosion(nucleus)
                contours = morphology.dilation(nucleus) & (~morphology.erosion(nucleus))

                aug_patch_boundary = contours.astype(np.uint8) * 1  # contours

                # aug_patch_boundary =boundary_npy
                aug_patch_tumor = aug_patch_seg.copy()

                aug_patch_msk = aug_patch_img.copy()
                aug_patch_msk[aug_patch_seg > 0] = 1

                img_ori_region = torch.from_numpy(np.expand_dims(aug_patch_img, 0)).unsqueeze(0)
                img_erase = torch.from_numpy(np.expand_dims(aug_patch_msk, 0)).unsqueeze(0)
                tumor_region = torch.from_numpy(np.expand_dims(aug_patch_tumor, 0)).unsqueeze(0)
                boundary_region = torch.from_numpy(np.expand_dims(aug_patch_boundary, 0)).unsqueeze(0)
                segment_region = torch.from_numpy(np.expand_dims(aug_patch_tumor, 0)).unsqueeze(0)
                batch = {
                    "img_erase": img_erase,
                    "img_ori": img_ori_region,
                    "tumor_region": tumor_region,
                    "boundary_region": boundary_region,
                    "img_seg": segment_region
                }
                batch_gen = syn.generate_batch(batch)
                batch_ret = syn.test_one_batch(batch_gen)
                fake_B = batch_ret['fake_B']
                psnr = cal_psnr(img_ori_region.detach().cpu().numpy(), fake_B.detach().cpu().numpy())
                ssim = cal_ssim(img_ori_region.detach().cpu().numpy(), fake_B.detach().cpu().numpy())
                mse = F.mse_loss(fake_B, batch_gen['real_B'])
                # fid=cal_fid_score()
                ssims.update(ssim)
                psnrs.update(psnr)
                mses.update(mse)
                # fids.update(fid)

                save_dir = syn.config.tmp_dir + '/test_epoch_' + str(epoch)
                if not isdir(save_dir):
                    os.makedirs(save_dir)
                visual_batch(batch_gen['real_A'], save_dir, "%s_%s_real_A" % (base_name, idx), channel=1, nrow=nrow)
                visual_batch(batch_gen['real_B'], save_dir, "%s_%s_real_B" % (base_name, idx), channel=1, nrow=nrow)
                visual_batch(batch_gen['boundary_B'], save_dir, "%s_%s_boundary_B" % (base_name, idx), channel=1,
                             nrow=nrow)
                visual_batch(batch_gen['tumor_B'], save_dir, "%s_%s_tumor_B" % (base_name, idx), channel=1,
                             nrow=nrow)
                visual_batch(torch.clamp(batch_ret['fake_B'] * 255 + 0.5, 0, 255), save_dir,
                             "%s_%s_gen_B" % (base_name, idx), channel=1,
                             nrow=nrow)
                if batch_ret['fake_B_coarse'] is not None:
                    visual_batch(torch.clamp(batch_ret['fake_B_coarse'] * 255 + 0.5, 0, 255), save_dir,
                                 "%s_gen_B_coarse" % (idx),
                                 channel=1,
                                 nrow=nrow)
                if batch_ret['boundary'] is not None:
                    visual_batch(torch.clamp(batch_ret['boundary'][-1] * 255 + 0.5, 0, 255), save_dir,
                                 "%s_gen_boundary" % (idx),
                                 channel=1,
                                 nrow=nrow)
                if batch_ret['boundary_coarse'] is not None:
                    visual_batch(torch.clamp(batch_ret['boundary_coarse'][-1] * 255 + 0.5, 0, 255), save_dir,
                                 "%s_gen_boundary_coarse" % (idx),
                                 channel=1,
                                 nrow=nrow)

                idx += 1
    zipDir(save_dir, save_dir + '.zip')
    shutil.rmtree(save_dir)

    # del loss, prec1
    empty_cache()
    # measure elapsed time
    info = 'Test Epoch Time {batch_time.avg:.3f}, '.format(batch_time=epoch_time) + \
           'SSIM {ssim.avg:f}, '.format(ssim=ssims) + \
           'PSNR {psnr.avg:f}, '.format(psnr=psnrs) + \
           'MSE {mse.avg:f}, '.format(mse=mses)
    # 'FID {fid.avg:f}, '.format(fid=losses_style)
    print(info)
    syn.log.flush()


def test_old(syn, test_dataset, epoch):
    syn.generator.eval()
    epoch_time = Averagvalue()
    ssims = Averagvalue()
    psnrs = Averagvalue()
    mses = Averagvalue()
    fids = Averagvalue()
    l2_loss = L2LossMaskedMean()
    nrow = 8
    with torch.no_grad():
        idx = 0
        for path in test_dataset.patches_imgs:
            base_name = os.path.basename(path)[:-4]
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
            psnr = cal_psnr(img_ori.detach().cpu().numpy(), fake_B.detach().cpu().numpy())
            ssim = cal_ssim(img_ori.detach().cpu().numpy(), fake_B.detach().cpu().numpy())
            mse = l2_loss(fake_B, batch_gen['real_B'], batch_gen['tumor_B'])
            # fid=cal_fid_score()
            ssims.update(ssim)
            psnrs.update(psnr)
            mses.update(mse)
            # fids.update(fid)

            save_dir = syn.config.tmp_dir + '/test_epoch_' + str(epoch)
            if not isdir(save_dir):
                os.makedirs(save_dir)
            visual_batch(batch_gen['real_A'], save_dir, "%s_%s_real_A" % (base_name, idx), channel=1, nrow=nrow)
            visual_batch(batch_gen['real_B'], save_dir, "%s_%s_real_B" % (base_name, idx), channel=1, nrow=nrow)
            visual_batch(batch_gen['boundary_B'], save_dir, "%s_%s_boundary_B" % (base_name, idx), channel=1,
                         nrow=nrow)
            visual_batch(batch_gen['tumor_B'], save_dir, "%s_%s_tumor_B" % (base_name, idx), channel=1,
                         nrow=nrow)
            visual_batch(torch.clamp(batch_ret['fake_B'] * 255 + 0.5, 0, 255), save_dir,
                         "%s_%s_gen_B" % (base_name, idx), channel=1,
                         nrow=nrow)
            if batch_ret['fake_B_coarse'] is not None:
                visual_batch(torch.clamp(batch_ret['fake_B_coarse'] * 255 + 0.5, 0, 255), save_dir,
                             "%s_%s_gen_B_coarse" % (base_name, idx),
                             channel=1,
                             nrow=nrow)
            if batch_ret['boundary'] is not None:
                visual_batch(torch.clamp(batch_ret['boundary'][-1] * 255 + 0.5, 0, 255), save_dir,
                             "%s_%s_gen_boundary" % (base_name, idx),
                             channel=1,
                             nrow=nrow)
            if batch_ret['boundary_coarse'] is not None:
                visual_batch(torch.clamp(batch_ret['boundary_coarse'][-1] * 255 + 0.5, 0, 255), save_dir,
                             "%s_%s_gen_boundary_coarse" % (base_name, idx),
                             channel=1,
                             nrow=nrow)

            idx += 1
        empty_cache()
    zipDir(save_dir, save_dir + '.zip')
    shutil.rmtree(save_dir)
    info = 'Test Epoch Time {batch_time.avg:.3f}, '.format(batch_time=epoch_time) + \
           'SSIM {ssim.avg:f}, '.format(ssim=ssims) + \
           'PSNR {psnr.avg:f}, '.format(psnr=psnrs) + \
           'MSE {mse.avg:f}, '.format(mse=mses)
    print(info)


def test_seg_old(seg_helper, test_dataset, epoch):
    seg_helper.model.eval()
    segmentation_metrics = {'dice': 0,
                            'jaccard': 0,
                            'voe': 0,
                            'rvd': 0,
                            'rmse': 0,
                            'mse': 0,
                            'hd': 0}
    lesion_segmentation_scores = {}
    nrow = 8
    with torch.no_grad():
        idx = 0
        for path in test_dataset.patches_imgs:
            base_name = os.path.basename(path)[:-4]
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
            real_A, real_B = seg_helper.generate_batch(batch)
            loss_s, acc_s, fakeB = seg_helper.test_one_batch(real_A, real_B)

            save_dir = seg_helper.config.tmp_dir + '/test_seg_epoch_' + str(epoch)
            if not isdir(save_dir):
                os.makedirs(save_dir)
            visual_batch(real_A, save_dir, "%s_%s_real_A" % (base_name, idx), channel=1, nrow=nrow)
            visual_batch(real_B, save_dir, "%s_%s_real_B" % (base_name, idx), channel=1, nrow=nrow)
            visual_batch(fakeB, save_dir, "%s_%s_seg_B" % (base_name, idx), channel=1,
                         nrow=nrow)
            idx += 1

            sliceFB = fakeB.detach().cpu().numpy()
            fake_B = np.squeeze(sliceFB, 1)
            fake_B = np.where(fake_B > 0.5, 1, 0)
            sliceRB = real_B.detach().cpu().numpy()
            real_B = np.squeeze(sliceRB, 1)
            fake_B = np.where(fake_B > 0.5, 1, 0)
            for j in range(fake_B.shape[0]):
                scores = compute_segmentation_scores(fake_B[j], real_B[j])
                scores2 = evaluate_metric(fake_B[j], real_B[j])
                for metric in segmentation_metrics:
                    if metric not in lesion_segmentation_scores:
                        lesion_segmentation_scores[metric] = []
                    if metric in scores:
                        lesion_segmentation_scores[metric].extend(scores[metric])
                    else:
                        lesion_segmentation_scores[metric].extend(scores2[metric])
            # if isTest and epoch > (seg.config.epochs - 2):
            #     sliceRA = real_A.detach().cpu().numpy()
            #     sliceRB = real_B.detach().cpu().numpy()
            #     sliceFB = fakeB.detach().cpu().numpy()
            #     fake_B = np.squeeze(sliceFB, 1)
            #     real_A = np.squeeze(sliceRA, 1)
            #     real_B = np.squeeze(sliceRB, 1)
            #     for b in range(fake_B.shape[0]):
            #         if b >= 4:
            #             continue
            #         save_dir = seg.config.tmp_dir + '/epoch_' + str(epoch) + '/batch_' + str(i) + '/' + str(b) + '/'
            #         if not isdir(save_dir):
            #             os.makedirs(save_dir)
            #         fake_B_numpy = fake_B[b]
            #         real_A_numpy = real_A[b]
            #         real_B_numpy = real_B[b]
            #         for sss in range(fake_B_numpy.shape[-1]):
            #             visualize(np.expand_dims(fake_B_numpy[..., sss], -1), join(save_dir, str(sss) + "Agen"))
            #             visualize(np.expand_dims(real_A_numpy[..., sss], -1), join(save_dir, str(sss) + "Binput"))
            #             visualize(np.expand_dims(real_B_numpy[..., sss], -1), join(save_dir, str(sss) + "Creal"))

        empty_cache()
        # measure elapsed time
    zipDir(save_dir, save_dir + '.zip')
    shutil.rmtree(save_dir)
    lesion_segmentation_metrics = {}
    info = 'Test '
    for m in lesion_segmentation_scores:
        lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
        info += (m + '({val:.3f})  '.format(val=lesion_segmentation_metrics[m]))
    print(info)
