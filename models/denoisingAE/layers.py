# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine import InputSpec
from keras.legacy import interfaces
from keras.layers.core import Layer, Dense
from keras.layers.convolutional import _Conv, Conv2DTranspose
from keras import backend as K

'''
@ Contents
    1. class: DependentDense
    2. class: DependentConv2DTranspose
'''

class DependentDense(Dense):
    """Dense layer having its trainable weights tied with its master layer."""
    def __init__(self, units,
                 master_layer,
                 layer_history_index=0,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        try:
            # Compatibility with the Functional API
            self._master_layer = master_layer._keras_history[
                layer_history_index
            ]
        except AttributeError:
            # For use with the Sequential API
            self._master_layer = master_layer
        super(DependentDense, self).__init__(units=units,
                                             activation=activation,
                                             use_bias=use_bias,
                                             kernel_initializer=kernel_initializer,
                                             bias_initializer=bias_initializer,
                                             kernel_regularizer=kernel_regularizer,
                                             bias_regularizer=bias_regularizer,
                                             activity_regularizer=activity_regularizer,
                                             kernel_constraint=kernel_constraint,
                                             bias_constraint=bias_constraint,
                                             **kwargs)

    def build(self, input_shape):
        # Tie weights with those of the master layer
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.kernel = K.transpose(self._master_layer.kernel)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.units, ),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def get_config(self):
        config = {'master_layer': self._master_layer}
        base_config = super(DependentDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DependentConv2DTranspose(Conv2DTranspose):
    @interfaces.legacy_deconv2d_support
    def __init__(self, filters,
                 kernel_size,
                 master_layer,
                 layer_history_index=0,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        try:
            # Compatibility for the functional API
            self._master_layer = master_layer._keras_history[
                layer_history_index
            ]
        except AttributeError:
            # For use with the Sequential API
            self._master_layer = master_layer

        super(DependentConv2DTranspose, self).__init__(filters=filters,
                                                       kernel_size=kernel_size,
                                                       strides=strides,
                                                       padding=padding,
                                                       data_format=data_format,
                                                       dilation_rate=dilation_rate,
                                                       activation=activation,
                                                       use_bias=use_bias,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer,
                                                       kernel_regularizer=kernel_regularizer,
                                                       bias_regularizer=bias_regularizer,
                                                       activity_regularizer=activity_regularizer,
                                                       kernel_constraint=kernel_constraint,
                                                       bias_constraint=bias_constraint,
                                                       **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' + str(4) +
                             '; Received input shape:' + str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] == None:
            raise ValueError("The channel dimension of the inputs should be defined. Found 'None'.")
        input_dim = input_shape[channel_axis]

        # Kernel dims: (rows, cols, input_ch, output_ch)
        # kernel_shape = self.kernel_size + (input_dim, self.filters)  # conv2d
        # kernel_shape = self.kernel_size + (self.filters, input_dim)  # conv2dtranspose
        self.kernel = self._master_layer.kernel
        if self.use_bias:
            # FIXME: Why not use 'self._master_layer.bias'?
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters, ),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def get_config(self):
        config = {'master_layer': self._master_layer}
        base_config = super(DependentConv2DTranspose, self).get_config()
        base_config.pop('dilation_rate')
        return dict(list(base_config.items()) + list(config.items()))
