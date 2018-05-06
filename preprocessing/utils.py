# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


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
        False


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
