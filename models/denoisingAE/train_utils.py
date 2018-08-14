# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import h5py
import random
import pickle
import itertools

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

"""
@: Contents
    1. func: get_available_processors
    2. func: get_single_pair_from_pkl
    3. func: get_single_pair_from_npy
    4. func: get_single_pair_from_h5
    5. func: generate_batches_from_directory
    6. func: get_steps_per_epoch
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


def get_channel_index(filepath, channel_type='buildings'):

    # Read channel names from file
    with open(filepath, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    # Splitting tokens
    splitters = ['[Protoss_Units]',
                 '[Protoss_Buildings]',
                 '[Terran_Units]',
                 '[Terran_Buildings]']

    # Make list of (split_token, channel_name) tuples
    pairs = []
    for l in lines:
        if l in splitters:
            key = l
            continue
        else:
            pairs.append((key, l))

    # Channel types to keep
    if channel_type == 'units':
        valid_keys = [splitters[0], splitters[2]]
    if channel_type == 'buildings':
        valid_keys = [splitters[1], splitters[3]]

    channel_index = []
    for i, pair in enumerate(pairs):
        if pair[0] in valid_keys:
            channel_index.append(i)
        else:
            continue

    return channel_index


def get_single_pair_from_pkl(filepath, fog=True, output_size=128):
    """Read a single sample, a dictionary of 2 scipy.sparse.csr_matrices, from .pkl file."""

    assert os.path.isfile(filepath)  # A full path must be provided
    with open(filepath, 'rb') as f:
        sample_dict = pickle.load(f)
        assert isinstance(sample_dict, dict)

    channel_index = get_channel_index('../../preprocessing/data/UNITS_PROTOSS_TERRAN.txt',
                                      channel_type='buildings')
    channel_index = np.array(channel_index, dtype=np.int8)

    if fog:  # input: fog, output: original
        x_fog = sample_dict.get('fog').toarray()
        x_original = sample_dict.get('original').toarray()

        assert int(np.sqrt(x_fog.shape[0])) == output_size
        assert int(np.sqrt(x_original.shape[0])) == output_size
        channel_size = x_fog.shape[-1]

        x_fog = x_fog.reshape((output_size, output_size, channel_size))
        x_original = x_original.reshape((output_size, output_size, channel_size))

        x_fog = x_fog[:, :, channel_index]
        x_original = x_original[:, :, channel_index]
        assert x_fog.shape == x_original.shape

        return x_fog.astype(np.float32), x_original.astype(np.float32)

    else:  # input: original, output: original
        x_original = sample_dict.get('original').toarray()
        assert int(np.sqrt(x_original.shape[0])) == output_size
        channel_size = x_original.shape[-1]
        x_original = x_original.reshape((output_size, output_size, channel_size))
        x_original = x_original[:, :, channel_index]

        return x_original.astype(np.float32), x_original.astype(np.float32)


def get_single_pair_from_npy(filepath, fog=True):
    """Read a single sample of 2 numpy arrays from .npy file."""
    assert os.path.isfile(filepath)  # A full path must be provided
    x_fog, x_original = np.load(filepath)
    if fog:
        return x_fog.astype(np.float32), x_original.astype(np.float32)
    else:
        return x_original.astype(np.float32), x_original.astype(np.float32)


def get_single_pair_from_h5(filepath, fog=True):
    """Read a single sample of 2 numpy arrays from .h5 file."""
    assert os.path.isfile(filepath)  # A full path must be provided
    with h5py.File(filepath, 'rb') as h5f:
        x_fog = h5f['fog'][:]
        x_original = h5f['original'][:]
    if fog:
        return x_fog.astype(np.float32), x_original.astype(np.float32)
    else:
        return x_original.astype(np.float32), x_original.astype(np.float32)


def generate_batches_from_directory(path_to_dir, file_format='pkl',
                                    start=None, end=None,
                                    batch_size=32, output_size=128):
    """Data generator to be used in the 'fit_generator' method."""
    # FIXME: one epoch = each sample is trained only once.

    if file_format == 'pkl':
        get_single_pair = get_single_pair_from_pkl
    elif file_format == 'npy':
        get_single_pair = get_single_pair_from_npy
    elif file_format == 'h5':
        get_single_pair = get_single_pair_from_h5
    else:
        raise ValueError

    assert batch_size % 2 == 0
    assert os.path.isdir(path_to_dir)

    filenames = os.listdir(path_to_dir)
    filenames = sorted(filenames)

    if (start is None) or (end is None):
        start = 0
        end = len(filenames)
    else:
        pass

    filenames = filenames[start:end]
    filenames = [os.path.join(path_to_dir, x) for x in filenames]  # relative path -> absolute path

    num_samples = len(filenames)
    steps_per_epoch = get_steps_per_epoch(num_samples, batch_size)

    while True:
        np.random.shuffle(filenames)  # shuffle every epoch
        for step in range(steps_per_epoch):

            batch_start = batch_size * step
            batch_end = min(batch_start + batch_size, num_samples - 1)

            filenames_batch = filenames[batch_start:batch_end]

            while True:
                try:
                    pairs_batch = [get_single_pair(filepath=x) for x in filenames_batch]
                    X_fog = [pair[0] for pair in pairs_batch]
                    X_fog = np.stack(X_fog, axis=0)
                    X_original = [pair[1] for pair in pairs_batch]
                    X_original = np.stack(X_original, axis=0)
                    break

                except Error as e:
                    continue

            yield (X_fog, X_original)


def get_steps_per_epoch(num_samples_in_epoch, batch_size):
    """
    Calculates number of batch iterations per epoch,
    necessary argument to be used in 'fit_generator'.
    """
    return math.ceil(num_samples_in_epoch / batch_size)
