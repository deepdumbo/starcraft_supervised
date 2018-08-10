# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime

import numpy as np
import tensorflow as tf

import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from ae import convolutional_encoder_decoder
from train_utils import generate_batches_from_directory, get_steps_per_epoch

num_classes = 2
batch_size = 32
epochs = 1
learning_rate = 0.001
shape = (128, 128, 49)

if __name__ == '__main__':

    model = convolutional_encoder_decoder(input_shape=shape)
    model = multi_gpu_model(model, gpus=4, cpu_merge=False)
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='mse')

    # Callbacks
    callbacks = []
    checkpoint_dir = './checkpoints/'
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    check = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'trained_dae.h5'),
                            monitor='loss',
                            save_best_only=True,
                            save_weights_only=True)
    callbacks.append(check)

    csv_logger = CSVLogger('./logs/training.log')
    callbacks.append(csv_logger)

    # Train
    train_dir = 'D:/parsingData/trainingData_v9/train/'
    test_dir  = 'D:/parsingData/trainingData_v9/test/'

    generator_train = generate_batches_from_directory(path_to_dir=train_dir,
                                                      file_format='pkl',
                                                      batch_size=batch_size)

    generator_valid = generate_batches_from_directory(path_to_dir=test_dir,
                                                      file_format='pkl',
                                                      batch_size=batch_size)

    steps_per_epoch_train = get_steps_per_epoch(len(os.listdir(train_dir)), batch_size)
    steps_per_epoch_valid = get_steps_per_epoch(len(os.listdir(test_dir)), batch_size)

    try:
        # FIXME: validation data results in error after single epoch
        model.fit_generator(generator=generator_train,
                            steps_per_epoch=steps_per_epoch_train,
                            epochs=epochs,
                            max_queue_size=batch_size * 4,
                            validation_data=generator_valid,
                            validation_steps=steps_per_epoch_valid,
                            callbacks=callbacks,
                            workers=6,
                            verbose=1)
    except KeyboardInterrupt:
        # Save model & weights
        now = datetime.datetime.now().strftime("%Y%m%d")
        savedir = './trained_models/{}/'.format(now)
        if not os.path.isdir(savedir):
            os.makedirs(savedir)
        single_model = [l for l in model.layers if l.name == 'model_1']
        assert len(single_model) == 1; single_model = single_model[0]
        single_model.save_weights(os.path.join(savedir, 'weights.h5'))
