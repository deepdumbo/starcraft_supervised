# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def cnn(input_tensor, available_devices=None):
    """
    Builds the convolution module necessary to build the CRNN model from the following paper:
        Title: An End-to-End Trainable Neural Network for Image-based Sequence
            Recognition and Its Application to Scene Text Recognition
        Link: https://arxiv.org/pdf/1507.05717.pdf
    """
    # TODO: Add support for multi device allocation of tensor variables
    assert input_tensor.shape.__len__() == 4
    # Block 1: [B, 128, 128, 25] -> [B, 64, 64, 32]
    block_1 = conv_block(input_tensor=input_tensor,
                         filters=[32, 32],
                         kernel_sizes=[3, 3],
                         strides=[2, 1],
                         block_name='block_1',
                        reuse=tf.AUTO_REUSE)
    # Block 2: [B, 64, 64, 32] -> [B, 32, 32, 64]
    block_2 = conv_block(input_tensor=block_1,
                         filters=[64, 64],
                         kernel_sizes=[3, 3],
                         strides=[2, 1],
                         block_name='block_2',
                         reuse=tf.AUTO_REUSE)
    # Block 3: [B, 32, 32, 64] -> [B, 16, 16, 128]
    block_3 = conv_block(input_tensor=block_2,
                         filters=[128, 128],
                         kernel_sizes=[3, 3],
                         strides=[2, 1],
                         block_name='block_3',
                         reuse=tf.AUTO_REUSE)
    # Block 4: [B, 16, 16, 128] -> [B, 8, 8, 256]
    block_4 = conv_block(input_tensor=block_3,
                         filters=[256, 256],
                         kernel_sizes=[3, 3],
                         strides=[2, 1],
                         block_name='block_4',
                         reuse=tf.AUTO_REUSE)
    output_tensor = global_pooling(block_4, pool_type='average')
    return output_tensor


def conv_block(input_tensor, filters, kernel_sizes, strides, block_name, reuse):
    """
    A VGG-style convolution block, refer to the following paper:
        Title: Simonyan et al., Very Deep Convolutional Networks for Large-Scale Image Recognition
        Link: https://arxiv.org/pdf/1409.1556.pdf
    """
    # TODO: Add batch normalization to block
    assert input_tensor.shape.__len__() == 4
    assert filters.__len__() == 2
    assert kernel_sizes.__len__() == 2
    assert strides.__len__() == 2
    with tf.variable_scope(block_name, reuse=reuse):
        conv_1 = tf.layers.conv2d(inputs=input_tensor,
                                  filters=filters[0],
                                  kernel_size=kernel_sizes[0],
                                  strides=strides[0],
                                  padding='same',
                                  activation=tf.nn.relu,
                                  use_bias=True,
                                  name='conv_1')
        conv_2 = tf.layers.conv2d(inputs=conv_1,
                                  filters=filters[1],
                                  kernel_size=kernel_sizes[1],
                                  strides=strides[1],
                                  padding='same',
                                  activation=tf.nn.relu,
                                  use_bias=True,
                                  name='conv_2')
        return conv_2


def global_pooling(input_tensor, pool_type):
    """Takes average/max on feature maps individually."""
    assert input_tensor.shape.__len__() == 4
    scope = 'global_{}_pooling'.format(pool_type)
    with tf.name_scope(scope):
        if pool_type == 'average':
            return tf.reduce_mean(input_tensor, axis=[1, 2])
        elif pool_type == 'max':
            return tf.reduce_max(input_tensor, axis=[1, 2])
        else:
            raise ValueError("Only 'average' and 'max' pooling is supported.")


if __name__ == '__main__':

    # Define hyperparameters
    num_classes = 2
    batch_size = None
    max_time = None
    input_shape = [128, 128, 25]
    learning_rate = 0.001

    # Define placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, max_time, ] + input_shape)
    Y = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_classes])

    # CNN part
    with tf.variable_scope('CNN', reuse=tf.AUTO_REUSE):
        cnn_inputs = tf.transpose(X, [1, 0, 2, 3, 4], name='batch2time')  # [B, T, H, W, C] -> [T, B, H, W, C]
        cnn_outputs = tf.map_fn(fn=lambda x: cnn(input_tensor=x),
                                elems=cnn_inputs,
                                name='conv_along_time_axis')
        cnn_outputs = tf.transpose(cnn_outputs, [1, 0, 2], name='time2batch')  # [T, B, output_size] -> [B, T, output_size]
        assert cnn_outputs.shape.__len__() == 3

    # RNN part
    rnn_inputs = cnn_outputs
    with tf.variable_scope('RNN', reuse=tf.AUTO_REUSE):
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128,
                                            name='lstm_cell_1')
        outputs, states = tf.nn.dynamic_rnn(cell=cell,
                                            inputs=rnn_inputs,
                                            time_major=False,
                                            dtype=tf.float32)

    # Fully-connected part
    with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
        fc = tf.contrib.layers.fully_connected(inputs=outputs[:, -1, :],
                                               num_outputs=64,
                                               activation_fn=tf.nn.relu)
        prediction = tf.contrib.layers.fully_connected(inputs=fc,
                                                       num_outputs=num_classes,
                                                       activation_fn=None)

    # Training part
    with tf.name_scope('train'):
        # Define loss operation
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=Y,
                                                       logits=prediction,
                                                       name='loss_fn')
        # Define gradient descent optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           name='optimizer')
        # Calculate gradients: 'compute_gradients()' returns a list of (gradient, variable) pairs
        grads_and_vars = optimizer.compute_gradients(loss=loss,
                                                     var_list=None)
        # Gradient update
        gradient_update = optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                                    name='gradient_update')

