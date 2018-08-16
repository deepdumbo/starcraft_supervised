# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns

file_dir = './reconstructed/20180813_000000/'
filenames = os.listdir(file_dir)

input_shape = (128, 128, 34)

i = 0
filename = filenames[i]

for i, filename in enumerate(filenames):

    with open(os.path.join(file_dir, filename), 'rb') as f:
        x_fog, x_original, x_pred = pickle.load(f)

    x_fog = x_fog.toarray().reshape((1, ) + input_shape)
    x_original = x_original.toarray().reshape((1, ) + input_shape)
    x_pred = x_pred.toarray().reshape((1, ) + input_shape)

    x_fog = np.squeeze(x_fog, axis=0)
    x_original = np.squeeze(x_original, axis=0)
    x_pred = np.squeeze(x_pred, axis=0)

    x_fog = np.transpose(x_fog, [2, 0, 1])
    x_original = np.transpose(x_original, [2, 0, 1])
    x_pred = np.transpose(x_pred, [2, 0, 1])

    plot_dir = './plots/'
    os.makedirs(plot_dir, exist_ok=True)
    thres = .5
    k = 0
    for f, o, p in zip(x_fog, x_original, x_pred):
        fig, axes = plt.subplots(1, 3, figsize=(40, 10))
        sns.heatmap(f, mask=(f<thres), ax=axes[0])
        sns.heatmap(o, mask=(o<thres), ax=axes[1])
        sns.heatmap(p, mask=(p<thres), ax=axes[2])
        fig.savefig('./plots/channels_{}.png'.format(k))
        k += 1

    break