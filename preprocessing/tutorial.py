# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import scipy.sparse as sp

filename = 'sample.pkl'

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
        print("{}: {}".format(k, v))

    # Get replay data
    replay_data = replay.get('replay_data')
    assert isinstance(replay_data, dict)
    print(">>> Number of keys (#. of samples): {}".format(replay_data.keys().__len__()))
    print(">>> Each key in 'replay_data' corresponds to a timestep, where its value is a list of two items.")
    print(">>> The first item is the sample in scipy.sparse.csr_matrix.")
    print(">>> The second item is a dictionary holding sample info.")


    # EXAMPLE 1: Get a 3D tensor, corresponding to a single frame (=one timestep)
    sample_index = 0
    assert sample_index in replay_data.keys()
    sample, sample_info = replay_data.get(sample_index)

    # Convert 'scipy.sparse.csr_matrix' back to a '2D numpy array'
    assert isinstance(sample, sp.csr_matrix)
    sample = sample.toarray()
    print(">>> Shape of sample in 2D: {}".format(sample.shape))

    # Reshape 2D numpy array back into a 3D numpy array
    assert isinstance(sample, np.ndarray)
    num_channels = sample.shape[-1]
    height, width = 128, 128
    sample = sample.reshape((height, width, num_channels))
    print(">>> A single sample is a 3D numpy array of shape {}".format(sample.shape))

    # Example 2: Get a 4D tensor, corresponding to a single training observation (=one replay)
    sequence_length = len(replay_data.keys())
    samples = [s for s, _ in replay_data.values()]  # Get first elements of 'replay_data' values
    samples = [s.toarray() for s in samples]             # Convert each sample to a 2D numpy array
    samples = [s.reshape((height, width, num_channels)) for s in samples]  # Reshape each sample to a 3D numpy array
    samples = np.stack(samples, axis=0)                  # list of 3D numpy arrays => 4D numpy array
    samples = samples.astype(np.float32)                 # change type for efficient usage of memory and speed
    print(">>> A training observation is a 3D numpy array of shape {}".format(samples.shape))
