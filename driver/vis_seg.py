# -*- coding: utf-8 -*-
# @Time    : 20/12/11 10:11
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : vis_seg.py

import matplotlib
import sys

sys.path.extend(["../../", "../", "./"])
from utils.utils import *
import torch
import random
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter

matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
base_root = '../log/eval_cls/'


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
    plt.rcParams['lines.markersize'] = 5


def main():
    lrs = ["1e-3", "1e-4", "1e-5"] * 3
    backbone_names = ['lr=1e-3', 'lr=1e-4', 'lr=1e-5']
    methods = [backbone_names[0]] * 3 + [backbone_names[1]] * 3 + [backbone_names[2]] * 3
    values = np.hstack([np.array([0.7566, 0.7610, 0.7390]),
                        np.array([0.8618, 0.8684, 0.8553]),
                        np.array([0.8136, 0.8092, 0.8000])])
    data_frame = pd.DataFrame(values.T,
                              columns=['values'],
                              )
    data_frame['backbone'] = methods
    data_frame['lrs'] = lrs
    plot_sns_line(data_frame, 'CLS_lrs')

    exit(0)


# ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

def plot_sns_line(data_frames, title):
    sns.set(font="serif")
    sns.set_style('whitegrid')
    fig=plt.figure(figsize=(8, 6))
    # fig, ax = plt.subplots(dpi=500,figsize=(8, 6))  # 9, 7
    # plt_props()
    sns.set_context('paper', font_scale=1.5, rc={"lines.linewidth": 1.7})
    ax = sns.lineplot(x=data_frames['lrs'], y=data_frames['values'], hue=data_frames['backbone'],
                      markers=True, dashes=["", "", "", "", "", "", "", "", ""], style=data_frames['backbone']
                      )
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_ylim(0.60, 0.90)
    plt.legend(loc=4)
    plt.legend(ncol=1, loc='lower right')
    fig.tight_layout()
    plt.xlabel('learning rate (lr)')
    plt.ylabel('ACC (%)')
    my_y_ticks = np.arange(0.60, 1.00, 0.1)
    plt.yticks(my_y_ticks)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(join(base_root, title + "_final.jpg"))
    plt.close()


if __name__ == '__main__':
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    random.seed(666)
    np.random.seed(666)
    main()
