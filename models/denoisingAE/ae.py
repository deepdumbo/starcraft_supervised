# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.backend as K

from keras.engine.training import Model
from keras.engine.topology import Input
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization

from layers import DependentDense
from layers import DependentConv2DTranspose

'''
@ Contents
    1. func: denoising AE(input_shape)
'''

def denoising_AE(input_shape):
    """Function to build a denoising autoencoder with symmetrically tied weights."""
    act = 'relu'
    use_bias = False

    # ENCODER: batch_shape: (B, 128, 128, 49)
    img_input = Input(shape=input_shape, name='input')

    # Block 1: (B, 128, 128, 49) -> (B, 64, 64, 64)
    B1_conv1 = Conv2D(64, 3, padding='same', use_bias=use_bias, name='B1_conv1')(img_input)
    B1_conv1_act = BatchNormalization(name='B1_bn1')(B1_conv1)
    B1_conv1_act = Activation(act, name='B1_act1')(B1_conv1_act)
    B1_conv2 = Conv2D(64, 3, padding='same', use_bias=use_bias, name='B1_conv2')(B1_conv1_act)
    B1_conv2_act = BatchNormalization(name='B1_bn2')(B1_conv2)
    B1_conv2_act = Activation(act, name='B1_act2')(B1_conv2_act)
    B1_conv2_pool = AveragePooling2D(name='B1_pool')(B1_conv2_act)

    # Block 2: (B, 64, 64, 64) -> (B, 32, 32, 128)
    B2_conv1 = Conv2D(128, 3, padding='same', use_bias=use_bias, name='B2_conv1')(B1_conv2_pool)
    B2_conv1_act = BatchNormalization(name='B2_bn1')(B2_conv1)
    B2_conv1_act = Activation(act, name='B2_act1')(B2_conv1_act)
    B2_conv2 = Conv2D(128, 3, padding='same', use_bias=use_bias, name='B2_conv2')(B2_conv1_act)
    B2_conv2_act = BatchNormalization(name='B2_bn2')(B2_conv2)
    B2_conv2_act = Activation(act, name='B2_act2')(B2_conv2_act)
    B2_conv2_pool = AveragePooling2D(name='B2_pool')(B2_conv2_act)

    # Block 3: (B, 32, 32, 128) -> (B, 16, 16, 256)
    B3_conv1 = Conv2D(256, 3, padding='same', use_bias=use_bias, name='B3_conv1')(B2_conv2_pool)
    B3_conv1_act = BatchNormalization(name='B3_bn1')(B3_conv1)
    B3_conv1_act = Activation(act, name='B3_act1')(B3_conv1_act)
    B3_conv2 = Conv2D(256, 3, padding='same', use_bias=use_bias, name='B3_conv2')(B3_conv1_act)
    B3_conv2_act = BatchNormalization(name='B3_bn2')(B3_conv2)
    B3_conv2_act = Activation(act, name='B3_act2')(B3_conv2_act)
    B3_conv3 = Conv2D(256, 3, padding='same', use_bias=use_bias, name='B3_conv3')(B3_conv2_act)
    B3_conv3_act = BatchNormalization(name='B3_bn3')(B3_conv3)
    B3_conv3_act = Activation(act, name='B3_act3')(B3_conv3_act)
    B3_conv3_pool = AveragePooling2D(name='B3_pool')(B3_conv3_act)

    # Block 4: (B, 16, 16, 256) -> (B, 4, 4, 512)
    B4_conv1 = Conv2D(512, 3, padding='same', use_bias=use_bias, name='B4_conv1')(B3_conv3_pool)
    B4_conv1_act = BatchNormalization(name='B4_bn1')(B4_conv1)
    B4_conv1_act = Activation(act, name='B4_act1')(B4_conv1_act)
    B4_conv2 = Conv2D(512, 3, padding='same', use_bias=use_bias, name='B4_conv2')(B4_conv1_act)
    B4_conv2_act = BatchNormalization(name='B4_bn2')(B4_conv2)
    B4_conv2_act = Activation(act, name='B4_act2')(B4_conv2_act)
    B4_conv3 = Conv2D(512, 3, padding='same', use_bias=use_bias, name='B4_conv3')(B4_conv2_act)
    B4_conv3_act = BatchNormalization(name='B4_bn3')(B4_conv3)
    B4_conv3_act = Activation(act, name='B4_act3')(B4_conv3_act)
    B4_conv3_pool = AveragePooling2D(pool_size=(4, 4), name='B4_pool')(B4_conv3_act)

    # Block 5: (B, 4, 4, 512) -> (B, 4 x 4 x 512)
    B5_flat = Flatten(name='B5_flatten')(B4_conv3_pool)
    B5_dense = Dense(1024, activation=act, name='feat')(B5_flat)

    # book keeping feature map size, with out batch dimension
    mid_shape = K.int_shape(B4_conv3_pool)[1:]  # (4, 4, 512)

    # Block 5': (B, 4 x 4 x 512) -> (B, 4, 4, 512)
    units = mid_shape[0] * mid_shape[1] * mid_shape[2]  # = 4 x 4 x 512
    B5_dense_prime = DependentDense(units, master_layer=B5_dense)(B5_dense)
    B5_flat_prime = Reshape(target_shape=(mid_shape))(B5_dense_prime)

    # Block 4': (B, 4, 4, 512) -> (B, 16, 16, 256)
    B4_conv3_pool_prime = UpSampling2D(size=(4, 4), name='B4_upsample_prime')(B5_flat_prime)
    B4_conv3_pool_prime = Activation(act, name='B4_upact_prime')(B4_conv3_pool_prime)
    B4_conv3_prime = DependentConv2DTranspose(
        512, 3, master_layer=B4_conv3, use_bias=use_bias, padding='same', name='B4_conv3_prime')(B4_conv3_pool_prime)
    B4_conv3_prime = Activation(act, name='B4_act3_prime')(B4_conv3_prime)
    B4_conv2_prime = DependentConv2DTranspose(
        512, 3, master_layer=B4_conv2, use_bias=use_bias, padding='same', name='B4_conv2_prime')(B4_conv3_prime)
    B4_conv2_prime = Activation(act, name='B4_act2_prime')(B4_conv2_prime)
    B4_conv1_prime = DependentConv2DTranspose(
        256, 3, master_layer=B4_conv1, use_bias=use_bias, padding='same', name='B4_conv1_prime')(B4_conv2_prime)
    B4_conv1_prime = Activation(act, name='B4_act1_prime')(B4_conv1_prime)

    # Block 3': (B, 16, 16, 256) -> (B, 32, 32, 128)
    B3_conv3_pool_prime = UpSampling2D(size=(2, 2), name='B3_upsample_prime')(B4_conv1_prime)
    B3_conv3_pool_prime = Activation(act, name='B3_upact_prime')(B3_conv3_pool_prime)
    B3_conv3_prime = DependentConv2DTranspose(
        256, 3, master_layer=B3_conv3, use_bias=use_bias, padding='same', name='B3_conv3_prime')(B3_conv3_pool_prime)
    B3_conv3_prime = Activation(act, name='B3_act3_prime')(B3_conv3_prime)
    B3_conv2_prime = DependentConv2DTranspose(
        256, 3, master_layer=B3_conv2, use_bias=use_bias, padding='same', name='B3_conv2_prime')(B3_conv3_prime)
    B3_conv2_prime = Activation(act, name='B3_act2_prime')(B3_conv2_prime)
    B3_conv1_prime = DependentConv2DTranspose(
        128, 3, master_layer=B3_conv1, use_bias=use_bias, padding='same', name='B3_conv1_prime')(B3_conv2_prime)
    B3_conv1_prime = Activation(act, name='B3_act1_prime')(B3_conv1_prime)

    # Block 2': (B, 32, 32, 128) -> (B, 64, 64, 64)
    B2_conv2_pool_prime = UpSampling2D(size=(2, 2), name='B2_upsample_prime')(B3_conv1_prime)
    B2_conv2_pool_prime = Activation(act, name='B2_upact_prime')(B2_conv2_pool_prime)
    B2_conv2_prime = DependentConv2DTranspose(
        128, 3, master_layer=B2_conv2, use_bias=use_bias, padding='same', name='B2_conv2_prime')(B2_conv2_pool_prime)
    B2_conv2_prime = Activation(act, name='B2_act2_prime')(B2_conv2_prime)
    B2_conv1_prime = DependentConv2DTranspose(
        64, 3, master_layer=B2_conv1, use_bias=use_bias, padding='same', name='B2_conv1_prime')(B2_conv2_prime)
    B2_conv1_prime = Activation(act, name='B2_act1_prime')(B2_conv1_prime)

    # Block 1': (B, 64, 64, 64) -> (B, 128, 128, 49)
    B1_conv2_pool_prime = UpSampling2D(size=(2, 2), name='B1_upsample_prime')(B2_conv1_prime)
    B1_conv2_pool_prime = Activation(act, name='B1_upact_prime')(B1_conv2_pool_prime)
    B1_conv2_prime = DependentConv2DTranspose(
        64, 3, master_layer=B1_conv2, use_bias=use_bias, padding='same', name='B1_conv2_prime')(B1_conv2_pool_prime)
    B1_conv2_prime = Activation(act, name='B1_act2_prime')(B1_conv2_prime)
    B1_conv1_prime = DependentConv2DTranspose(
        input_shape[-1], 3, master_layer=B1_conv1, use_bias=use_bias, padding='same', name='B1_conv1_prime')(B1_conv2_prime)
    recon = Activation(act, name='recon')(B1_conv1_prime)

    model = Model(inputs=[img_input], outputs=[B5_dense, recon])
    return model
