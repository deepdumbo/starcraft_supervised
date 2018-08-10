# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

from tqdm import tqdm
from logger import create_logger

"""
@: Contents
    1. func: get_my_id(replay_info, token)
    2. func: apply_fog_of_war(sample, sample_info, me, output_size=128)
    3. main
"""

player_name = '박성균'
output_size = 128
data_dir = 'Y:/trainingData_v9/data(replay)/{}/{}/'.format(player_name, output_size)

save_formats = ['pkl', 'npy', 'h5']
save_format = save_formats[0]
write_dir = 'Y:/trainingData_v9/data(sample)/{}'.format(player_name)


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
    '''Mask values with zero for opponent's units not yet exposed.\n'''
    assert isinstance(sample, sp.csr_matrix)
    assert isinstance(sample_info, dict)
    assert me in [0, 1]

    channel_names = sample_info.get('channel_names')
    df = sample_info.get('dataframe').copy()
    df['getName'] = sample_info.get('getName')

    original_size = 4096
    channel_size = len(channel_names)
    assert int(np.sqrt(sample.shape[0])) == output_size

    x_fog, x_original = sample.toarray().copy(), sample.toarray().copy()
    x_fog = x_fog.reshape((output_size, output_size, channel_size))
    x_original = x_original.reshape((output_size, output_size, channel_size))

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

    assert (x_fog.shape == x_original.shape)
    assert isinstance(x_fog, np.ndarray) and isinstance(x_original, np.ndarray)

    x_fog = x_fog.reshape(sample.shape)
    x_fog = sp.csr_matrix(x_fog.astype(np.uint8))
    x_original = x_original.reshape(sample.shape)
    x_original = sp.csr_matrix(x_original.astype(np.uint8))
    assert isinstance(x_fog, sp.csr_matrix) and isinstance(x_original, sp.csr_matrix)

    return x_fog, x_original


def apply_fog_of_war_v2(sample, sample_info, me, output_size=128):
    # TODO: Implement this function, check feasibility.
    raise NotImplementedError(
        "Exposed units which are not visible should use the last seen coordinates."
    )


if __name__ == '__main__':

    # Create logger instance
    logger = create_logger(name='fog_of_war', level='INFO', stream=True,
                           outfile='fog_{}_{}'.format(player_name, save_format))

    # Get list of pickle replay files
    filenames = os.listdir(data_dir)
    logger.info("Parsing {} replays to sample-level pkl files...".format(len(filenames)))

    for i, filename in enumerate(filenames):

        # Open pickle file (1 pickle = 1 replay)
        with open(os.path.join(data_dir, filename), 'rb') as f:
            replay_data, replay_info = pickle.load(f).values()
            assert isinstance(replay_data, dict)
            assert isinstance(replay_info, dict)

        num_samples = len(replay_data.keys())

        # Get my playerID in current replay
        me = get_my_id(replay_info, token='Terran')
        assert me in [0, 1]

        # Iterate over frames, make (x_fog, x_original) pairs, save to file
        for k, (sample, sample_info) in tqdm(replay_data.items()):
            assert isinstance(sample, sp.csr_matrix)
            assert isinstance(sample_info, dict)
            x_fog, x_original = apply_fog_of_war(sample=sample,
                                                 sample_info=sample_info,
                                                 me=me,
                                                 output_size=output_size)

            fname = filename.split('.')[0]
            if not os.path.isdir(write_dir):
                os.makedirs(write_dir)

            # Save method 1: save dictionary of two csr_matrices to .pkl file
            if save_format == 'pkl':
                writefile = '{}_{}_over_{}.pkl'.format(fname, k, num_samples)
                with open(os.path.join(write_dir, writefile), 'wb') as f:
                    # Save samples as sparse matrices (for memory issues)
                    x_fog = x_fog.reshape((-1, x_fog.shape[-1]))
                    x_original = x_original.reshape((-1, x_original.shape[-1]))
                    result = {'fog': x_fog,
                              'original': x_original}
                    pickle.dump(result, f)

            # Save method 2: save 2 numpy arrays to .npy file
            elif save_format == 'npy':
                writefile = '{}_{}_over_{}.npy'.format(fname, k, num_samples)
                np.save(writefile, (x_fog, x_original))

            # Save method 3: save 2 numpy arrays to .h5 file
            elif save_format == 'h5':
                writefile = '{}_{}_over_{}.h5'.format(fname, k, num_samples)
                with h5py.File(os.path.join(write_dir, writefile), 'wb') as h5f:
                    h5f.create_dataset('fog', data=x_fog)
                    h5f.create_dataset('original', data=x_original)

            else:
                raise ValueError("file format '{}' not supported.".format(save_format))


        logger.info("[{:>4}/{:>4}] Parsed replay to {} samples, with fog-of-war.".format(
            i + 1, len(filenames), num_samples
        ))
