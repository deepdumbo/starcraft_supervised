# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import math
import random
import pickle
import asyncio
import collections

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import keras
import keras.backend as K

from tensorflow.python.client import device_lib
from keras.engine.training import Model
from keras.engine.topology import Input
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.recurrent import LSTM
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.wrappers import TimeDistributed

num_classes = 2
batch_size = 4
epochs = 100
learning_rate = 0.001

def get_available_processors(only_gpus=True):
    """Get the names of available processors"""
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        local_device_protos = device_lib.list_local_devices()
        if only_gpus:
            return [d.name for d in local_device_protos if d.device_type == 'GPU']
        else:
            return [d.name for d in local_device_protos]


def get_array_from_replay(filepath, maxlen=3):
    """Given filepath, return replay as a 4D numpy array."""
    assert os.path.isfile(filepath)  # A full path must be provided
    with open(filepath, 'rb') as f:
        replay = pickle.load(f)
        assert isinstance(replay, dict)
    replay_data = replay.get('replay_data')

    samples = [s for s, _ in replay_data.values()]
    samples = [s.toarray() for s in samples]

    output_size = math.sqrt(samples[0].shape[0]).__int__()
    num_channels = samples[0].shape[1]

    samples = [s.reshape((output_size, output_size, num_channels)) for s in samples]
    samples = np.stack(samples, axis=0)
    samples = samples.astype(np.float32)

    # Pad or truncate
    # TODO: Change argument 'maxlen'
    seq_length = samples.shape[0]
    if seq_length < maxlen:
        pad_length = maxlen - seq_length
        zeros = np.zeros((pad_length, ) + samples.shape[1:])
        samples = np.concatenate((zeros, samples))
    elif seq_length > maxlen:
        samples = samples[-maxlen:, :, :, :]
    else:
        pass
    assert samples.shape == ((maxlen, ) + samples.shape[1:])
    return samples


def generate_arrays_from_directory(path_to_dir, batch_size):
    """Data generator to be used in the 'fit_generator' method."""
    assert batch_size % 2 == 0
    assert os.path.isdir(path_to_dir)
    filenames = os.listdir(path_to_dir)
    win_names = [x for x in filenames if x.split('_')[0] == '1']
    win_names = [os.path.join(path_to_dir, x) for x in win_names]
    lose_names = [x for x in filenames if x.split('_')[0] == '0']
    lose_names = [os.path.join(path_to_dir, x) for x in lose_names]
    steps_per_epoch = math.ceil(len(filenames) / batch_size)
    while True:
        for step in range(steps_per_epoch):
            # FIXME: Change map function to list comprehension style
            win_batch = list(map(get_array_from_replay,
                                 random.sample(win_names, batch_size // 2)
                                 )
                             )
            # FIXME: Change map function to list comprehension style
            lose_batch = list(map(get_array_from_replay,
                                  random.sample(lose_names, batch_size // 2)
                                  )
                              )
            X = np.concatenate((win_batch, lose_batch), axis=0)
            Y = np.zeros(shape=(batch_size, 2))

            y = [1] * (batch_size // 2) + [0] * (batch_size // 2)
            Y = keras.utils.to_categorical(y, num_classes=len(set(y)))
            yield (X, Y)


if __name__ == '__main__':

    # Check that we have 4 available gpus
    available_devices = get_available_processors(only_gpus=True)
    if available_devices.__len__() == 1:
        available_devices = [available_devices[0] for _ in range(4)]
        assert available_devices.__len__() == 4
    else:
        assert available_devices.__len__() == 4

    with tf.device(available_devices[0]):
        # Tensor shape: (B, T, 128, 128, 49) -> (B, T, 64, 64, 64)
        cnn_input = Input(batch_shape=(None, None, 128, 128, 49))
        y = TimeDistributed(Conv2D(64, 3, padding='same', activation='relu'))(cnn_input)
        y = TimeDistributed(Conv2D(64, 3, padding='same', activation='relu'))(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(y)

    with tf.device(available_devices[1]):
        # Tensor shape: (B, T, 64, 64, 64) -> (B, T, 32, 32, 128)
        y = TimeDistributed(Conv2D(128, 3, padding='same', activation='relu'))(y)
        y = TimeDistributed(Conv2D(128, 3, padding='same', activation='relu'))(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(y)

    with tf.device(available_devices[2]):
        # Tensor shape: (B, T, 32, 32, 128) -> (B, T, 16, 16, 256)
        y = TimeDistributed(Conv2D(256, 3, padding='same', activation='relu'))(y)
        y = TimeDistributed(Conv2D(256, 3, padding='same', activation='relu'))(y)
        y = TimeDistributed(Conv2D(256, 3, padding='same', activation='relu'))(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(y)

        # Tensor shape: (B, T, 16, 16, 256) -> (B, T, 8, 8, 512)
        y = TimeDistributed(Conv2D(512, 3, padding='same', activation='relu'))(y)
        y = TimeDistributed(Conv2D(512, 3, padding='same', activation='relu'))(y)
        y = TimeDistributed(Conv2D(512, 3, padding='same', activation='relu'))(y)
        y = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(y)

        # Tensor shape: (B, T, 4, 4, 512) -> (B, T, 1, 512)
        y = TimeDistributed(GlobalAveragePooling2D())(y)

    with tf.device(available_devices[3]):
        y = LSTM(128, return_sequences=False, return_state=False)(y)
        prediction = Dense(num_classes, activation='softmax')(y)

    # Instantiate a 'keras.engine.training.Model' instance
    model = Model(inputs=cnn_input, outputs=prediction)

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    # Train model
    datadir = 'Z:/1. 프로젝트/2018_삼성SDS_스타크래프트/Supervised/trainingData_v3/data(선수별)/박성균/128/'
    generator = generate_arrays_from_directory(datadir, batch_size)
    steps_per_epoch = math.ceil(os.listdir(datadir).__len__() / batch_size)
    model.fit_generator(generator=generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        max_queue_size=10,
                        verbose=1)
