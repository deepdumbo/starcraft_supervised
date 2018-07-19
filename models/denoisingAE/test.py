# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import tensorflow as tf
import keras
import scipy.sparse as sp

from tqdm import tqdm
from ae import denoising_AE
from train_utils import get_single_pair_from_npy


if __name__ == "__main__":

    model = denoising_AE(input_shape=(128, 128, 49))
    model.load_weights('./trained_models/20180719/weights_epochs_1.h5')

    data_dir = 'D:/parsingData/parsingData_v4/by_sample_npy/'
    filelist = os.listdir(data_dir)

    for i, file in tqdm(enumerate(filelist)):
        x_fog, x_original = get_single_pair_from_npy(os.path.join(data_dir, file), denoising=True)
        x_pred = model.predict(np.expand_dims(x_fog, axis=0))
        x_pred = np.squeeze(x_pred, 0)

        x_fog = sp.csr_matrix(x_fog.reshape((-1, 49)))
        x_original = sp.csr_matrix(x_original.reshape((-1, 49)))
        x_pred = sp.csr_matrix(x_pred.reshape((-1, 49)))

        writedir = './reconstructed/'
        if not os.path.isdir(writedir):
            os.makedirs(writedir)
        with open(writedir + file.split('.')[0] + '_reconstructed.pkl', 'wb') as f:
            pickle.dump([x_fog, x_original, x_pred], f)

        if i + 1 == 10000:
            break
