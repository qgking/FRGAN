# -*- coding: utf-8 -*-
# @Time    : 20/12/11 10:06
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : vis_test.py
import matplotlib
import sys

sys.path.extend(["../../", "../", "./"])
from utils.utils import *
import torch
import random
import matplotlib
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import pickle
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import f1_score, average_precision_score

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
base_root = "../data_root/patches/"
import pandas as pd


def plt_props():
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.style'] = 'normal'
    plt.rcParams['font.variant'] = 'normal'
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['figure.figsize'] = 6, 4
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 7


def main():
    nomal_aug = 'kidney_tumor_syn_tumor_aug_normal'  # 'kidney_tumor_syn_tumor_aug_normal' #'liver_tumor_syn_tumor_aug_normal'
    data_name = 'KiTS19'  # 'LiTS17'  'KiTS19'
    experiments = [nomal_aug, 'noneofall_SYNHelper_None',
                   'dilate_SYNHelper_None', 'gatedilate_SYNHelper_None',
                   'gatedilatericher_SYNHelper_Dilate_Gate_Richer',
                   'gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lcon',
                   'gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lcon_Lpercep',
                   'gatedilatericher_SYNHelper_Dilate_Gate_Richer_Lcon_Lpercep_Lsty']
    names = experiments
    old_data = []
    for ex in range(len(experiments)):
        if 'aug_normal' in experiments[ex]:
            syn_data_set_dir = base_root + experiments[ex]
            syn_datalist = sorted(glob(join(syn_data_set_dir, "*.npy")),
                                  reverse=False)
        else:
            syn_data_set_dir = base_root + data_name + "/" + experiments[ex]
            syn_datalist = sorted(glob(join(syn_data_set_dir, "*.npy")),
                                  reverse=False)
        print("dataloaded %s" % (syn_data_set_dir))
        syn_data = []
        print("Total patch len %s" % (len(syn_datalist)))
        for syn_file in syn_datalist:
            img = np.load(join(syn_file))
            img_ori = np.expand_dims(img[..., 0], 0)
            syn_data.append(img_ori)
        syn_data = np.array(syn_data)
        print(syn_data.shape)
        syn_data = syn_data.reshape(syn_data.shape[0] *
                                    syn_data.shape[1] * syn_data.shape[2] * syn_data.shape[3] * syn_data.shape[4])
        print(syn_data.shape)
        print("Name %s" % (names[ex]))
        print("mean %.3f" % (
            np.mean(syn_data)))
        print("std %.3f" % (
            np.std(syn_data)))
        if ex > 0:
            stc = stats.ttest_ind(old_data, syn_data,equal_var = False)
            print(stc)
        old_data = syn_data
        print(20 * "-")
    exit(0)


if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)
    main()
