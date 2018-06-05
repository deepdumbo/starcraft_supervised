# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

"""
@: Contents
    1. func: get_available_processors
    2. func: get_single_pair
    3. func: generate_train_batches_from_directory
    4. func: get_steps_per_epoch
"""


def get_available_processors(only_gpus=True):
    """Get the names of available processors"""
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        local_device_protos = device_lib.list_local_devices()
        if only_gpus:
            return [d.name for d in local_device_protos if d.device_type == 'GPU']
        else:
            return [d.name for d in local_device_protos]


def get_single_pair(filepath, output_size=128):
    """
    Function only to be used in a generator.\n
    Arguments:\n
        filepath: absolute path to .pkl file, of the following format;
            i.e. path/to/dir/0_PWTL_박성균_72_b799996378ccea7369e02749015b5839dd03bf6d_128_0_over_2.pkl
    Returns:
        (x_fog, x_original) pairs, each of shape (W, H, C).\n
    """
    assert os.path.isfile(filepath)  # A full path must be provided
    with open(filepath, 'rb') as f:
        sample_dict = pickle.load(f)
        assert isinstance(sample_dict, dict)

    x_fog = sample_dict.get('fog').toarray()
    x_original = sample_dict.get('fog').toarray()

    assert int(np.sqrt(x_fog.shape[0])) == output_size
    assert int(np.sqrt(x_original.shape[0])) == output_size
    channel_size = x_fog.shape[-1]  # equivalent to x_original.shape[-1]

    x_fog = x_fog.reshape((output_size, output_size, channel_size))
    x_original = x_original.reshape((output_size, output_size, channel_size))
    assert x_fog.shape == x_original.shape

    return (x_fog.astype(np.float32), x_original.astype(np.float32))


def generate_train_batches_from_directory(path_to_dir, batch_size, output_size=128):
    """Data generator to be used in the 'fit_generator' method."""
    assert batch_size % 2 == 0
    assert os.path.isdir(path_to_dir)

    filenames = os.listdir(path_to_dir)
    filenames = sorted(filenames)

    win_names = [x for x in filenames if x.split('_')[0] == '1']
    win_names = [os.path.join(path_to_dir, x) for x in win_names]
    lose_names = [x for x in filenames if x.split('_')[0] == '0']
    lose_names = [os.path.join(path_to_dir, x) for x in lose_names]
    steps_per_epoch = get_steps_per_epoch(len(filenames), batch_size)

    while True:
        for step in range(steps_per_epoch):

            win_batch = [
                get_single_pair(w, output_size) for w in (random.sample(win_names, batch_size // 2))
            ]
            lose_batch =[
                get_single_pair(l, output_size) for l in (random.sample(lose_names, batch_size // 2))
            ]

            total_batch = win_batch + lose_batch

            X_fog =[pair[0] for pair in total_batch]
            X_fog = np.stack(X_fog, axis=0)

            X_original = [pair[1] for pair in total_batch]
            X_original = np.stack(X_original, axis=0)

            yield (X_fog, X_original)


def get_steps_per_epoch(num_samples_in_epoch, batch_size):
    """
    Calculates number of batch iterations per epoch,
    necessary argument to be used in 'fit_generator'.
    """
    return math.ceil(num_samples_in_epoch / batch_size)
