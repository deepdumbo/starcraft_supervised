# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

"""
@: Contents
    1. func: get_my_id(replay_info, token)
    2. func: apply_fog_of_war(sample, sample_info, me, output_size=128)
    3. main
"""

player_name = '박성균'
output_size = 128
path_to_dir = 'D:/parsingData/trainingData_v4/data(선수별)/{}/{}/'.format(player_name, output_size)
writedir = 'D:/parsingData/trainingData_v4/by_sample/'

def get_my_id(replay_info, token):
    """
    Get my 'playerId' from information available from 'replay_info'.
    Games against the same tribes are not yet supported.\n
    Arguments:\n
        replay_info: dict, holding replay information.\n
        token: str, indicating my tribe, from the following:
         ('T', 'P', 'Z') or ('Terran', 'Protoss', 'Zerg').\n
    Returns:\n
        int, 0 or 1 indicating 'playerId'.
    """
    assert isinstance(replay_info, dict)
    assert isinstance(token, str)

    if len(token) > 1:
        token = token[0].upper()
    else:
        token = token.upper()
    versus = replay_info.get('versus')  # i.e. 'PWTL'
    p0, p1 = versus[0], versus[2]       # i.e. 'P', 'T'

    if p0 == p1:
        raise NotImplementedError("Replays with the same tribes are not yet supported.")
    else:
        # match with 'playerId' column
        if p0 == token:
            me = 0
        elif p1 == token:
            me = 1
        else:
            raise ValueError("No Terrans detected in replay.")
    return me


def apply_fog_of_war(sample, sample_info, me, output_size=128):
    """Mask values with zero for opponent's units not yet exposed."""
    assert isinstance(sample, sp.csr_matrix)
    assert isinstance(sample_info, dict)
    assert me in [0, 1]

    channel_names = sample_info.get('channel_names')
    df = sample_info.get('dataframe').copy()
    df['getName'] = sample_info.get('getName')

    original_size = 4096
    channel_size = len(channel_names)
    assert int(np.sqrt(sample.shape[0])) == output_size

    x_fog, x_original = sample.toarray().copy(), sample
    x_fog = x_fog.reshape((output_size, output_size, channel_size))

    for i, unit in df.iterrows():
        player = unit['playerId']
        exposed = unit['isExposed']
        pos_x = unit['getPosition.x']
        pos_y = unit['getPosition.y']
        gn = unit['getName']
        # uid = unit['unitId']
        # attacking = unit['isAttacking']
        # visible = unit['isVisible']
        if (player == me) or (gn not in channel_names):
            # My units are always visible, or
            # the current unit is not in channel, OUT OF CONSIDERATION
            continue
        else:
            if exposed == 1:
                # Opponent's unit already exposed
                continue
            elif exposed == 0:
                # Opponent's unit not yet exposed
                ch_index = channel_names.index(gn)
                pos_x = int(pos_x * output_size / original_size)
                pos_y = int(pos_y * output_size / original_size)
                x_fog[pos_x, pos_y, ch_index] = 0  # Mask zeros

    x_fog = x_fog.reshape((-1, channel_size))
    return sp.csr_matrix(x_fog), x_original


if __name__ == '__main__':
    filenames = os.listdir(path_to_dir)
    for filename in filenames:

        # Open pickle file
        with open(os.path.join(path_to_dir, filename), 'rb') as f:
            replay_data, replay_info = pickle.load(f).values()
            assert isinstance(replay_data, dict)
            assert isinstance(replay_info, dict)

        num_samples = replay_data.keys().__len__()

        # Get my playerID in current replay
        me = get_my_id(replay_info, token='Terran')
        assert me in [0, 1]

        # Iterate over frames, make (x_fog, x_original) pairs, save to file
        for k, (sample, sample_info) in replay_data.items():
            x_fog, x_original = apply_fog_of_war(sample, sample_info, me, output_size)
            result = {'fog': x_fog,
                      'original': x_original,
                      'frame_count': k}

            filename_ = filename.split('.')[0]
            if not os.path.isdir(writedir):
                os.makedirs(writedir)
            writefile = '{}_{}_over_{}.pkl'.format(filename_, k, num_samples)
            with open(os.path.join(writedir, writefile), 'wb') as f:
                pickle.dump(result, f)

        print(">>> {} has been parsed at a sample-level with fog-of-war; {} samples.".format(
            filename, num_samples)
        )
