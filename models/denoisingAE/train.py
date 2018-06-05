# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math

import numpy as np
import tensorflow as tf

import keras
from keras.engine.training import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from ae import denoising_AE
from train_utils import get_available_processors, generate_arrays_from_directory

num_classes = 2
batch_size = 16
epochs = 100
learning_rate = 0.001
shape = (128, 128, 49)

if __name__ == '__main__':

    # FIXME: is it L2-norm?
    base_model = denoising_AE(input_shape=shape)
    base_model.compile(optimizer=Adam(lr=learning_rate),
                       loss={'recon': 'mse', 'feat': 'mse'})

    model = Model(inputs=[base_model.input], outputs=[base_model.outputs[-1]])
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='mse')

    # TODO: Add more callbacks
    callbacks = []
    check_dir = 'D:/parsingData/checkpoints/'
    if not os.path.isdir(check_dir):
        os.makedirs(check_dir)
    check = ModelCheckpoint(filepath=os.path.join(check_dir, 'trained_AE.h5'),
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True)
    callbacks.append(check)

    datadir = 'D:/parsingData/trainingData_v4/by_sample/'
    generator = generate_arrays_from_directory(datadir, batch_size)
    steps_per_epoch = math.ceil(os.listdir(datadir).__len__() / batch_size)
    model.fit_generator(generator=generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        max_queue_size=64,
                        callbacks=callbacks,
                        verbose=1)
