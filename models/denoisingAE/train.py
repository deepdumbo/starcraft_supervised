# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime

import numpy as np
import tensorflow as tf

import keras
import keras.backend as K

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from ae import convolutional_encoder_decoder
from train_utils import generate_batches_from_directory
from train_utils import get_steps_per_epoch

batch_size = 128
epochs = 30
learning_rate = 0.001
input_shape = (128, 128, 34)


def save(multi_gpu_model, checkpoint_dir, save_multi=True, save_single=True):

    # Load best checkpoint weights
    model.load_weights(
        filepath=os.path.join(checkpoint_dir, 'trained_weights.h5')
    )

    if save_multi:
        # Save weights to weight directory (multi-gpu model)
        weight_dir = './weights/'
        os.makedirs(os.path.join(weight_dir, 'multi/'), exist_ok=True)
        model.save_weights(
            filepath=os.path.join(weight_dir, 'multi_gpu_weights_{}.h5'.format(now))
        )
        print('>>> Saved multi-gpu model weights...')

    if save_single:
        # Save weights to weight directory (single-gpu model)
        single_model = [l for l in model.layers if l.name == 'model_1']
        single_model = single_model[0]
        os.makedirs(os.path.join(weight_dir, 'single/'), exist_ok=True)
        single_model.save_weights(
            filepath=os.path.join(weight_dir, 'single_gpu_weights_{}.h5'.format(now))
        )
        print('>>> Saved single-gpu model weights...')


if __name__ == '__main__':

    # TODO: Allow GPU usage growth

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model = convolutional_encoder_decoder(input_shape=input_shape)
    model = multi_gpu_model(model, gpus=4, cpu_merge=False)
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss='mse')

    # Callbacks
    callbacks = []
    checkpoint_dir = './checkpoints/{}'.format(now)
    os.makedirs(checkpoint_dir, exist_ok=True)
    check = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'trained_weights.h5'),
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=True)
    callbacks.append(check)

    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)
    csv_logger = CSVLogger(os.path.join(log_dir, 'train.log'))
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
    steps_per_epoch_valid = get_steps_per_epoch(len(os.listdir(test_dir)), int(batch_size / 2))

    try:
        # FIXME: use fit_on_batch and 'keras.Sequence' instance
        history = model.fit_generator(generator=generator_train,
                                      steps_per_epoch=steps_per_epoch_train,
                                      epochs=epochs,
                                      max_queue_size=batch_size * 4,
                                      validation_data=generator_valid,
                                      validation_steps=steps_per_epoch_valid,
                                      callbacks=callbacks,
                                      workers=6,
                                      verbose=1)
    except KeyboardInterrupt:

        try:
            # TODO: save model architecture as well (use CustomObjectScope)
            save(multi_gpu_model=model,
                 save_multi=True,
                 save_single=True)
        except FileNotFoundError as e:
            print(str(e))

    # Save model weights
    save(multi_gpu_model=model,
         save_multi=True,
         save_single=True)
