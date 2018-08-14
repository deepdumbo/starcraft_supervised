# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

import keras
import keras.backend as K

from tqdm import tqdm

from ae import convolutional_encoder_decoder
from train_utils import get_single_pair_from_pkl

input_shape = (128, 128, 34)
flattened_dim = input_shape[0] * input_shape[1] * input_shape[2]

if __name__ == "__main__":

    # Load model weights (on single-gpu)
    weight_dir = './weights/single/'
    weight_file = os.listdir(weight_dir)[-1]
    model = convolutional_encoder_decoder(input_shape=input_shape)
    model.load_weights(filepath=os.path.join(weight_dir, weight_file))

    test_dir = 'D:/parsingData/trainingData_v9/test/'
    filenames = os.listdir(test_dir)

    for i, filename in enumerate(tqdm(filenames)):
        x_fog, x_original = get_single_pair_from_pkl(os.path.join(test_dir, filename), fog=True)
        x_pred = model.predict(np.expand_dims(x_fog, axis=0))
        x_pred = np.squeeze(x_pred, 0)

        x_fog = sp.csr_matrix(x_fog.reshape((-1, flattened_dim)))
        x_original = sp.csr_matrix(x_original.reshape((-1, flattened_dim)))
        x_pred = sp.csr_matrix(x_pred.reshape((-1, flattened_dim)))

        # ex) weight_file = 'single_gpu_weights_20180813_000000.h5'
        writedir = './reconstructed/{}'.format('_'.join(weight_file.split('.')[0].split('_')[-2:]))
        os.makedirs(writedir, exist_ok=True)

        with open(writedir + filename.split('.')[0] + '_reconstructed.pkl', 'wb') as f:
            pickle.dump([x_fog, x_original, x_pred], f)
