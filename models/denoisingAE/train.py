# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import keras
from keras.engine.training import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from ae import denoising_AE
from train_utils import generate_train_batches_from_directory, get_steps_per_epoch

num_classes = 2
batch_size = 16
epochs = 100
learning_rate = 0.001
shape = (128, 128, 49)

if __name__ == '__main__':

    base_model = denoising_AE(input_shape=shape)
    inputs = [base_model.input]
    outputs = [x for x in base_model.outputs if x.name.startswith('recon')]  # use only decoder output
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='mse')

    # TODO: Add more callbacks
    callbacks = []
    checkpoint_dir = 'D:/parsingData/checkpoints/'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    check = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'trained_AE.h5'),
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True)
    callbacks.append(check)

    data_dir = 'D:/parsingData/trainingData_v4/by_sample/'
    generator = generate_train_batches_from_directory(data_dir, batch_size)
    steps_per_epoch = get_steps_per_epoch(len(os.listdir(data_dir)), batch_size)
    model.fit_generator(generator=generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        max_queue_size=64,
                        callbacks=callbacks,
                        verbose=1)
