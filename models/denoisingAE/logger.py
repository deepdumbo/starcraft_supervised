# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

def create_logger(name, level, stream=True, outfile=None):
    """
    Helper function which creates a logger provided its name.
    """
    # Create a logger instance
    logger = logging.getLogger(name)
    # Create a log formatter
    formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
    # Add stream log handlers
    if stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)
    # Add file log handlers
    if outfile != None:
        logdir = './logs'
        if not os.path.isdir(os.path.abspath(logdir)):
            os.makedirs(logdir)
        fileHandler = logging.FileHandler(os.path.join(logdir, outfile + '.log'))
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
    # Set level of logger
    if level is 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif level is 'INFO':
        logger.setLevel(logging.INFO)
    elif level is 'WARNING':
        logger.setLevel(logging.WARNING)
    elif level is 'CRITICAL':
        logger.setLevel(logging.CRITICAL)

    return logger
