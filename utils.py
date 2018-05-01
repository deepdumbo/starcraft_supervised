# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def get_filenames(filedir, logger=None):
    """Add function docstring."""
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
    token = against[0].upper()
    if token in versus:
        return True
    else:
        return False
