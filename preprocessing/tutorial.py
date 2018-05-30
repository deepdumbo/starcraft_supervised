# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import scipy.sparse as sp
import pandas as pd

from utils import custom_resize

filename = '../data/sample.pkl'

if __name__ == '__main__':

    # Load replay
    with open(filename, 'rb') as f:
        replay = pickle.load(f)

    assert isinstance(replay, dict)
    print(">>> The replay file is a dictionary with the following keys: '{}', '{}'.".format(*replay.keys()))

    # Get replay info
    replay_info = replay.get('replay_info')
    assert isinstance(replay_info, dict)
    print(">>> The replay info is a dictionary with the following keys:\n")
    for k, v in replay_info.items():
        print('{}: {}'.format(k, type(v)))

    # Get replay data
    replay_data = replay.get('replay_data')
    assert isinstance(replay_data, dict)
    print(">>> Number of keys (#. of samples): {}".format(replay_data.keys().__len__()))
    print(">>> Each key in 'replay_data' corresponds to a timestep, where its value is a list of two items.")
    print(">>> The first item is the sample in 'scipy.sparse.csr_matrix' format.")
    print(">>> The second item is a dictionary holding sample info.")


    # TODO: EXAMPLE 1: Get a 3D tensor, corresponding to a single frame (=one timestep)
    sample_index = 0
    assert sample_index in replay_data.keys()
    sample, sample_info = replay_data.get(sample_index)

    # Convert 'scipy.sparse.csr_matrix' back to a '2D numpy array'
    assert isinstance(sample, sp.csr_matrix)
    sample = sample.toarray()
    print(">>> Shape of sample in 2D: {}".format(sample.shape))

    # Reshape 2D numpy array back into a 3D numpy array
    assert isinstance(sample, np.ndarray)
    original_size = int(np.sqrt(sample.shape[0]))
    num_channels = sample.shape[-1]
    sample = sample.reshape((original_size, original_size, num_channels))
    print(">>> A single sample is a 3D numpy array of shape {} (original)".format(sample.shape))

    # Resize each feature map with a desired height and width
    assert len(sample.shape) == 3
    output_size = int(original_size / 2)
    sample = custom_resize(sample, output_size=output_size)
    print(">>> A single sample is a 3D numpy array of shape {} (resized)".format(sample.shape))


    # TODO: EXAMPLE 2: Get a 4D tensor, corresponding to a single training observation (=one replay)
    sequence_length = len(replay_data.keys())
    samples = [s for s, _ in replay_data.values()]       # Get first elements of 'replay_data' values
    samples = [s.toarray() for s in samples]             # Convert each sample to a 2D numpy array
    samples = [s.reshape((original_size, original_size, num_channels)) for s in samples]  # Reshape each as 3D
    samples = [custom_resize(s, output_size) for s in samples]  # downsize each sample to desired heights and widths
    samples = np.stack(samples, axis=0)                  # list of 3D numpy arrays => 4D numpy array
    samples = samples.astype(np.float32)                 # change type for efficient usage of memory and speed
    print(">>> A training observation is a 4D numpy array of shape {}".format(samples.shape))

    # In our case, RNNs will require 5-dimensional inputs of shape (B, T, H, W, C)
    samples = samples.reshape((1, ) + samples.shape)
    print(">>> A training observation is a 5D numpy array of shape {}".format(samples.shape))


    # TODO: Example 3: Fog-of-war
    sample_index = 28
    assert sample_index in replay_data.keys()
    sample, sample_info = replay_data.get(sample_index)

    df = sample_info.get('dataframe')
    df['getName'] = sample_info.get('getName')
    assert isinstance(df, pd.DataFrame)

    versus = replay_info.get('versus')  # i.e. 'PWTL'
    p0 = versus[0]                      # i.e. 'P'
    p1 = versus[2]                      # i.e. 'T'

    if p0 == p1:
        raise NotImplementedError("Games with the same tribes are not yet supported.")
    else:
        if p0 == 'T':
            me = 0
        elif p1 == 'T':
            me = 1

    channel_names = sample_info.get('channel_names')
    channel_mask = np.zeros(shape=(num_channels, ))
    for i, (user, gn, vis) in enumerate(zip(df['playerId'], df['getName'], df['isVisible'])):
        # 0. Skip 'getName's that are not in channels
        if gn not in channel_names:
            continue
        else:
            if user == me:
                # My units should always be visible
                ch_index = channel_names.index(gn)
                channel_mask[ch_index] = 1
            else:
                if vis == 1:
                    # Some opponent's units are visible
                    ch_index = channel_names.index(gn)
                    channel_mask[ch_index] = 1
                elif vis == 0:
                    # Opponent's units not visible
                    ch_index = channel_names.index(gn)
                    channel_mask[ch_index] = 0
                else:
                    pass

    assert channel_mask.shape[0] == channel_names.__len__()
