#coding=utf8
from __future__ import print_function
from __future__ import absolute_import
# import backend as K
import numpy as np
import keras
from keras import backend as K
from keras.engine import Layer
from keras.engine import InputSpec
from keras.layers import Dense,multiply,Add
import tensorflow as tf

class crossNetLayer(Layer):
    """ Return the outputs and last_output
    """
    def __init__(self,  **kwargs):
        super(crossNetLayer, self).__init__(**kwargs)
        self.supports_masking = True
    def build(self, input_shape):
        # Used purely for shape validation.
        self.input_dim = input_shape[2]
        self.W = self.add_weight(name='kernel',
                                 shape=(self.input_dim, 1),
                                 initializer='uniform',
                                 trainable=True)
        self.bias = self.add_weight(shape=(self.input_dim,),
                                    initializer='uniform',
                                    name='bias',
                                    trainable=True)
        super(crossNetLayer, self).build(input_shape)

    def call(self, input):
        input_t = K.reshape(input,shape=[-1,input.shape[1],input.shape[2],1])
        input_trans = K.reshape(input,shape=[-1,input.shape[1],1,input.shape[2]])
        value = K.dot(input_trans, self.W)
        cross = tf.matmul(input_t, value)
        result = K.reshape(cross,(-1,input.shape[1],input.shape[2]))
        result = K.bias_add(result, self.bias)+input
        return result

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None

