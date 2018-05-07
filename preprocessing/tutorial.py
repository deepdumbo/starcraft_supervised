# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import deepdish

import numpy as np
import scipy.sparse as sp

filename = 'PWTL_0_9413c3e460c69a7c3488c1c7da228d83357f1ebf.h5'

if __name__ == '__main__':

    # Load replay
    replay = deepdish.io.load(filename)
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
    print(">>> Each key in replay data corresponds to a specific time, where its corresponding value is a list of two items.")
    print(">>> The first item is the sample in scipy.sparse.csr_matrix, and the second item is a dictionary holding sample info.")

    # Get sample data
    sample_index = 0
    assert sample_index in replay_data.keys()
    sample, sample_info = replay_data.get(sample_index)

    # Convert 'scipy.sparse.csr_matrix' back to a '2D numpy array'
    assert isinstance(sample, sp.csr_matrix)
    sample = sample.toarray()

    # Reshape 2D numpy array back into a 3D numpy array
    assert isinstance(sample, np.ndarray)
    num_channels = sample.shape[-1]
    height, width = 128, 128
    sample = sample.reshape((height, width, num_channels))
    print(">>> A single sample is a 3D numpy array of shape {}".format(sample.shape))
