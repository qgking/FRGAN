from torch.utils import data
import numpy as np
import os
from glob import glob
import torch
from os.path import join, isdir
from utils.utils import visualize
import time

TMP_DIR = "./tmp"
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)
SYNDATA = 'SYN'
SEGDATA = 'SEG'
DATATYPES = [SYNDATA, SEGDATA]
from utils.utils import *


class SYNDataLoader(data.Dataset):
    """
    Dataloader
    """

    def __init__(self, root, split, data_type,
                 split_radio,
                 config):
        self.root = root
        self.split = split
        self.data_type = data_type
        self.config = config
        if data_type not in DATATYPES:
            raise ValueError("not support data provider!")

        # self.filelist = sorted(glob(join(self.root, "img_" + "*.npy")), reverse=False)
        # if self.split == 'train':
        #     self.patches_imgs = self.filelist[0:int(len(self.filelist) * split_radio)]
        # elif split == "vali":
        #     vali_radio = (1 - split_radio) / 2
        #     self.patches_imgs = self.filelist[
        #                         int(len(self.filelist) * split_radio):int(
        #                             len(self.filelist) * (split_radio + vali_radio))]
        # elif split == "test":
        #     test_radio = (1 - split_radio) / 2
        #     self.patches_imgs = self.filelist[
        #                         int(len(self.filelist) * (split_radio + test_radio)):]
        # else:
        #     raise ValueError("Invalid split type!")
        # print(split + " data size: " + str(len(self.patches_imgs)))

        self.filelist = sorted(glob(join(self.root, "img_" + "*.npy")), reverse=False)
        st_idx = self.filelist[0].rfind('img_') + 4
        end_edx = st_idx + 3
        train_file_names = [nn[st_idx:end_edx] for nn in self.filelist]
        nns = set()
        for nn in train_file_names:
            nns.add(nn)
        nns = sorted(nns, reverse=False)
        self.img_idx = nns
        print('Total volumes %d ' % (len(nns)))
        print('Total data patches %d ' % (len(self.filelist)))

        if self.split == 'train':
            self.patches_imgs_idx = self.img_idx[0:int(len(self.img_idx) * split_radio)]
        elif split == "vali":
            split_radio = 0.6  # TODO fix split radio valid and test
            vali_radio = (1 - split_radio) / 2
            self.patches_imgs_idx = self.img_idx[
                                    int(len(self.img_idx) * split_radio):int(
                                        len(self.img_idx) * (split_radio + vali_radio))]
        elif split == "test":
            split_radio = 0.6  # TODO fix split radio valid and test
            test_radio = (1 - split_radio) / 2
            self.patches_imgs_idx = self.img_idx[
                                    int(len(self.img_idx) * (split_radio + test_radio)):]
        else:
            raise ValueError("Invalid split type!")
        print(split + " image volume size: %d" % len(self.patches_imgs_idx), self.patches_imgs_idx)
        self.patches_imgs = [path for path in self.filelist if path[st_idx:end_edx] in self.patches_imgs_idx]
        print(split + " total data patch size: " + str(len(self.patches_imgs)))
        lab_set = open(join(self.config.tmp_dir, '%s_set_syn.txt' % (split)), 'w')
        for ll in range(len(self.patches_imgs)):
            lab_set.write(
                "%s" % (self.patches_imgs[ll]))
            lab_set.write("\n")

    def __len__(self):
        return len(self.patches_imgs)

    def __getitem__(self, index):
        img_file = self.patches_imgs[index]
        img = np.load(join(img_file))

        img_ori = torch.from_numpy(np.expand_dims(img[..., 0], 0))
        img_erase = torch.from_numpy(np.expand_dims(img[..., 1], 0))
        tumor_region = torch.from_numpy(np.expand_dims(img[..., 2], 0))
        boundary_region = torch.from_numpy(np.expand_dims(img[..., 3], 0))
        seg = torch.from_numpy(np.expand_dims(img[..., 4], 0))
        # for sss in range(img.shape[-2]):
        #     visualize(np.transpose(img_ori.detach().cpu().numpy(), (1, 2, 3, 0))[:, :, sss, :],
        #               join(TMP_DIR, str(sss) + "Araw"))
        #     visualize(np.transpose(img_erase.detach().cpu().numpy(), (1, 2, 3, 0))[:, :, sss, :],
        #               join(TMP_DIR, str(sss) + "Berase"))
        #     visualize(np.transpose(tumor_region.detach().cpu().numpy(), (1, 2, 3, 0))[:, :, sss, :],
        #               join(TMP_DIR, str(sss) + "Ctumor"))
        #     visualize(np.transpose(boundary_region.detach().cpu().numpy(), (1, 2, 3, 0))[:, :, sss, :],
        #               join(TMP_DIR, str(sss) + "Dboundary"))
        #     visualize(np.transpose(seg.detach().cpu().numpy(), (1, 2, 3, 0))[:, :, sss, :],
        #               join(TMP_DIR, str(sss) + "Eseg"))
        if self.data_type == SEGDATA:
            return {
                "img_ori": img_ori,
                "img_seg": seg,
                "img_erase": img_erase
            }
        else:
            return {
                "img_erase": img_erase,
                "img_ori": img_ori,
                "tumor_region": tumor_region,
                "boundary_region": boundary_region,
                "img_seg": seg
            }


class SEGDataLoader(data.Dataset):
    """
    Dataloader
    """

    def __init__(self, filelist, config,
                 split):
        self.patches_imgs = filelist
        self.config = config
        self.split = split
        print("data size: " + str(len(self.patches_imgs)))
        lab_set = open(join(self.config.tmp_dir, '%s_set_seg.txt' % (split)), 'w')
        for ll in range(len(self.patches_imgs)):
            lab_set.write(
                "%s" % (self.patches_imgs[ll]))
            lab_set.write("\n")

    def __len__(self):
        return len(self.patches_imgs)

    def __getitem__(self, index):
        img_file = self.patches_imgs[index]
        img = np.load(join(img_file))
        img_ori = torch.from_numpy(np.expand_dims(img[..., 0], 0))
        img_erase = torch.from_numpy(np.expand_dims(img[..., 1], 0))
        seg = torch.from_numpy(np.expand_dims(img[..., 4], 0))
        base_name = os.path.basename(img_file)[:-4]
        # visual_batch(img_ori.unsqueeze(0), self.config.tmp_dir, "%s_syn_A" % (base_name), channel=1, nrow=8)
        # visual_batch(seg.unsqueeze(0), self.config.tmp_dir, "%s_seg_A" % (base_name), channel=1, nrow=8)

        return {
            "img_ori": img_ori,
            "img_seg": seg,
            "img_erase": img_erase
        }
