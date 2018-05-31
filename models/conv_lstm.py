# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import collections

import numpy as np

import tensorflow as tf

from tensorflow.contrib.rnn.python.ops.rnn_cell import Conv2DLSTMCell

# Defined for a single timestep
batch_size = None
max_time = None
input_shape = [128, 128, 25]

with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE) as vs:
    # Define a cell
    # TODO: add support for padding='valid'
    cell = Conv2DLSTMCell(input_shape=input_shape,
                          output_channels=2,
                          kernel_shape=[7, 7],
                          use_bias=True,
                          name='conv_2d_lstm_cell_1')
    # Define input placeholder
    x_input = tf.placeholder(dtype=tf.float32,
                             shape=[batch_size, max_time, ] + input_shape,
                             name='rnn_input')
    # Define rnn layer
    outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                       inputs=x_input,
                                       time_major=False,
                                       dtype=tf.float32,
                                       scope=vs)
    assert outputs.shape.__len__() == 5
    
    # Switch batch & time dimensions [B, T, ...] => [T, B, ...]
    outputs = tf.transpose(outputs, [1, 0, 2, 3, 4])
    outputs = tf.map_fn(fn=lambda x: tf.nn.max_pool(value=x, 
                                                    ksize=[1, 2, 2, 1], 
                                                    strides=[1, 2, 2, 1],
                                                    padding='VALID'),
                        elems=outputs,
                        name='maxpool_along_time')
    # Switch time & batch dimensions [T, B, ...] => [B, T, ...]
    outputs = tf.transpose(outputs, [1, 0, 2, 3, 4])
       