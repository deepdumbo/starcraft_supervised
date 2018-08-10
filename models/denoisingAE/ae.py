# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
import keras.backend as K

from keras.engine.training import Model
from keras.engine.topology import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from layers import DependentConv2DTranspose

'''
@ Contents
    1. func: convolutional_encoder_decoder(input_shape)
'''


def convolutional_encoder_decoder(input_shape, use_bias=False):
    """Build a convolutional encoder-decoder with symmetrically tied weights."""
    # FIXME: add dilated convolution
    # FIXME: mask?

    # Define input tensor, batch_shape: (B, 128, 128, C)
    img_input = Input(shape=input_shape, name='input')

    # ENCODER:
    with K.name_scope('encoder'):

        # Block 1: (B, 128, 128, C) -> (B, 64, 64, 64)
        with K.name_scope('block1') as ns:
            b1_conv1 = Conv2D(filters=64, kernel_size=3, strides=1,
                              padding='same', use_bias=use_bias, name=ns+'conv1')(img_input)
            b1_conv1_act = BatchNormalization(name=ns+'bn1')(b1_conv1)
            b1_conv1_act = LeakyReLU(name=ns+'act1')(b1_conv1_act)
            b1_conv2 = Conv2D(filters=64, kernel_size=3, strides=2,
                              padding='same', use_bias=use_bias, name=ns+'conv2')(b1_conv1_act)
            b1_conv2_act = BatchNormalization(name=ns+'bn2')(b1_conv2)
            b1_conv2_act = LeakyReLU(name=ns+'act2')(b1_conv2_act)
            b1_out = Lambda(lambda x: x, name=ns+'out')(b1_conv2_act)

        # Block 2: (B, 64, 64, 64) -> (B, 32, 32, 128)
        with K.name_scope('block2') as ns:
            b2_conv1 = Conv2D(filters=128, kernel_size=3, strides=1,
                              padding='same', use_bias=use_bias, name=ns+'conv1')(b1_out)
            b2_conv1_act = BatchNormalization(name=ns+'bn1')(b2_conv1)
            b2_conv1_act = LeakyReLU(name=ns+'act1')(b2_conv1_act)
            b2_conv2 = Conv2D(filters=128, kernel_size=3, strides=2,
                              padding='same', use_bias=use_bias, name=ns+'conv2')(b2_conv1_act)
            b2_conv2_act = BatchNormalization(name=ns+'bn2')(b2_conv2)
            b2_conv2_act = LeakyReLU(name=ns+'act2')(b2_conv2_act)
            b2_out = Lambda(lambda x: x, name=ns+'out')(b2_conv2_act)

        # Block 3: (B, 32, 32, 128) -> (B, 16, 16, 256)
        with K.name_scope('block3') as ns:
            b3_conv1 = Conv2D(filters=256, kernel_size=3, strides=1,
                              padding='same', use_bias=use_bias, name=ns+'conv1')(b2_out)
            b3_conv1_act = BatchNormalization(name=ns+'bn1')(b3_conv1)
            b3_conv1_act = LeakyReLU(name=ns+'act1')(b3_conv1_act)
            b3_conv2 = Conv2D(filters=256, kernel_size=3, strides=1,
                              padding='same', use_bias=use_bias, name=ns+'conv2')(b3_conv1_act)
            b3_conv2_act = BatchNormalization(name=ns+'bn2')(b3_conv2)
            b3_conv2_act = LeakyReLU(name=ns+'act2')(b3_conv2_act)
            b3_conv3 = Conv2D(filters=256, kernel_size=3, strides=2,
                              padding='same', use_bias=use_bias, name=ns+'conv3')(b3_conv2_act)
            b3_conv3_act = BatchNormalization(name=ns+'bn3')(b3_conv3)
            b3_conv3_act = LeakyReLU(name=ns+'act3')(b3_conv3_act)
            b3_out = Lambda(lambda x: x, name=ns+'out')(b3_conv3_act)

        # Block 4: (B, 16, 16, 256) -> (B, 8, 8, 512)
        with K.name_scope('block4') as ns:
            b4_conv1 = Conv2D(filters=512, kernel_size=3, strides=1,
                              padding='same', use_bias=use_bias, name=ns+'conv1')(b3_out)
            b4_conv1_act = BatchNormalization(name=ns+'bn1')(b4_conv1)
            b4_conv1_act = LeakyReLU(name=ns+'act1')(b4_conv1_act)
            b4_conv2 = Conv2D(filters=512, kernel_size=3, strides=1,
                              padding='same', use_bias=use_bias, name=ns+'conv2')(b4_conv1_act)
            b4_conv2_act = BatchNormalization(name=ns+'bn2')(b4_conv2)
            b4_conv2_act = LeakyReLU(name=ns+'act2')(b4_conv2_act)
            b4_conv3 = Conv2D(filters=512, kernel_size=3, strides=2,
                              padding='same', use_bias=use_bias, name=ns+'conv3')(b4_conv2_act)
            b4_conv3_act = BatchNormalization(name=ns+'bn3')(b4_conv3)
            b4_conv3_act = LeakyReLU(name=ns+'act3')(b4_conv3_act)
            b4_out = Lambda(lambda x: x, name=ns+'out')(b4_conv3_act)

    with K.name_scope('decoder'):

        # Block 4': (B, 8, 8, 512) -> (B, 16, 16, 256)
        with K.name_scope('block4_t') as ns:
            b4_convt3 = DependentConv2DTranspose(filters=512, kernel_size=3, strides=2, master_layer=b4_conv3,
                                                  padding='same', use_bias=use_bias, name=ns+'convt3')(b4_out)
            b4_convt3_act = LeakyReLU(name=ns+'act3')(b4_convt3)
            b4_convt2 = DependentConv2DTranspose(filters=512, kernel_size=3, strides=1, master_layer=b4_conv2,
                                                 padding='same', use_bias=use_bias, name=ns+'convt2')(b4_convt3_act)
            b4_convt2_act = LeakyReLU(name=ns+'act2')(b4_convt2)
            b4_convt1 = DependentConv2DTranspose(filters=256, kernel_size=3, strides=1, master_layer=b4_conv1,
                                                 padding='same', use_bias=use_bias, name=ns+'convt1')(b4_convt2_act)
            b4_convt1_act = LeakyReLU(name=ns+'act1')(b4_convt1)
            b4_t_out = Lambda(lambda x: x, name=ns+'out')(b4_convt1_act)

        # Block 3': (B, 16, 16, 256) -> (B, 32, 32, 128)
        with K.name_scope('block3_t') as ns:
            b3_convt3 = DependentConv2DTranspose(filters=256, kernel_size=3, strides=2, master_layer=b3_conv3,
                                                 padding='same', use_bias=use_bias, name=ns+'convt3')(b4_t_out)
            b3_convt3_act = LeakyReLU(name=ns+'act3')(b3_convt3)
            b3_convt2 = DependentConv2DTranspose(filters=256, kernel_size=3, strides=1, master_layer=b3_conv2,
                                                 padding='same', use_bias=use_bias, name=ns+'convt2')(b3_convt3_act)
            b3_convt2_act = LeakyReLU(name=ns+'act2')(b3_convt3_act)
            b3_convt1 = DependentConv2DTranspose(filters=128, kernel_size=3, strides=1, master_layer=b3_conv1,
                                                 padding='same', use_bias=use_bias, name=ns+'convt1')(b3_convt2_act)
            b3_convt1_act = LeakyReLU(name=ns+'act1')(b3_convt1)
            b3_t_out = Lambda(lambda x: x, name=ns+'out')(b3_convt1_act)

        # Block 2': (B, 32, 32, 128) -> (B, 64, 64, 64)
        with K.name_scope('block2_t') as ns:
            b2_convt2 = DependentConv2DTranspose(filters=128, kernel_size=3, strides=2, master_layer=b2_conv2,
                                                 padding='same', use_bias=use_bias, name=ns+'convt2')(b3_t_out)
            b2_convt2_act = LeakyReLU(name=ns+'act2')(b2_convt2)
            b2_convt1 = DependentConv2DTranspose(filters=64, kernel_size=3, strides=1, master_layer=b2_conv1,
                                                 padding='same', use_bias=use_bias, name=ns+'convt1')(b2_convt2_act)
            b2_convt1_act = LeakyReLU(name=ns+'act1')(b2_convt1)
            b2_t_out = Lambda(lambda x: x, name=ns+'out')(b2_convt1_act)

        # Block 1': (B, 64, 64, 64) -> (B, 128, 128, 49)
        with K.name_scope('block1_t') as ns:
            b1_convt2 = DependentConv2DTranspose(filters=64, kernel_size=3, strides=2, master_layer=b1_conv2,
                                                 padding='same', use_bias=use_bias, name=ns+'convt2')(b2_t_out)
            b1_convt2_act = LeakyReLU(name=ns+'act2')(b1_convt2)
            b1_convt1 = DependentConv2DTranspose(filters=input_shape[-1], kernel_size=3, strides=1, master_layer=b1_conv1,
                                                 padding='same', use_bias=use_bias, name=ns+'convt1')(b1_convt2_act)
            b1_convt1_act = LeakyReLU(name=ns+'act1')(b1_convt1)
            b1_t_out = Lambda(lambda x: x, name=ns+'out')(b1_convt1_act)

    img_output = Lambda(lambda x: x, name='output')(b1_t_out)
    model = Model(inputs=[img_input], outputs=[img_output])

    return model


if __name__ == '__main__':

    # Clear current graph
    K.clear_session()

    # Define input shape
    input_shape = (128, 128, 66)

    # Instantiate model
    model = convolutional_encoder_decoder(input_shape=input_shape)

    # Show model summary
    print(model.summary())