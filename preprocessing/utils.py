# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np


def get_filenames(filedir, logger=None):
    """
    Get all filenames under given file directory.\n
    Arguments:\n
        filedir: str, absolute path to directory.\n
        logger: a logger instance, for efficient logging.\n
    Returns:\n
        result: list of strings indicating the filenames.
    """
    filedir = os.path.abspath(filedir)
    if logger is not None:
        logger.info(
            "Will be parsing csv files from '{}'".format(filedir)
        )
    result = os.listdir(filedir)
    return result


def filter_filenames(versus, against):
    """
    Filter filenames.\n
    Arguments:\n
        versus: str, of the format, i.e. 'TLPW', 'TWPL', 'PLTW', 'PWTL'.\n
        against: str, one of 'terran', 'protoss', 'zerg'.\n
    Returns:\n
        boolean, whether to use filename or not.\n
    """
    a = against[0].upper()
    if all(x in versus for x in ['T', a]):
        return True
    else:
        return False


def get_game_result(versus, against):
    """
    Get the game result.\n
    Arguments:\n
        versus: str, of the format, i.e. 'TLPW', 'TWPL', 'PLTW', 'PWTL'.\n
        against: str, one of 'terran', 'protoss', 'zerg'.\n
    Returns:\n
        1 if we win, 0 otherwise.\n
    """
    a = against[0].upper()
    if versus == 'TW{}L'.format(a):
        return 1
    elif versus == 'TL{}W'.format(a):
        return 0
    elif versus == '{}WTL'.format(a):
        return 0
    elif versus == '{}LTW'.format(a):
        return 1


def custom_resize(A, output_size):
    """
    Resize height and width of 3D numpy array.\n
    Arguments:\n
        A: a 3D numpy array of shape (height, width, num_channels)
        output_size, int, only downsizing is supported.\n
    Returns:\n
        A downsized 3D numpy array of shape (height, width, num_channels)
    """
    assert isinstance(A, np.ndarray)
    if len(A.shape) == 3:
        assert A.shape[0] == A.shape[1]
        original_size = A.shape[0]
        num_channels = A.shape[-1]
        assert output_size <= original_size, "The 'output_size' must be smaller than the 'original_size'."
        ratio = output_size / original_size

        A = A.transpose((2, 0, 1))
        B = np.zeros(shape=(num_channels, output_size, output_size))

        for k in range(A.shape[0]):
            for i in range(A.shape[1]):
                i_ = int(i * ratio)
                for j in range(A.shape[2]):
                    j_ = int(j * ratio)
                    B[k, i_, j_] += A[k, i, j]

        # Transpose 3D numpy array back to shape of (height, width, num_channels)
        B = B.transpose((1, 2, 0))
        assert B.shape == (output_size, output_size, num_channels)
        return B
    else:
        raise ValueError(
            "Shape mismatch, expected 3D numpy array, received shape of {}.".format(A.shape)
        )
