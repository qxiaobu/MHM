#coding=utf8
from __future__ import print_function
from __future__ import absolute_import
# import backend as K
import numpy as np
import keras
from keras import backend as K
from keras.engine import Layer
from keras.engine import InputSpec
from keras.layers import GRU,Permute,Dense,multiply


class AttentionStep(Layer):
    """ Return the outputs and last_output
    """
    def __init__(self, units, dropout=0., **kwargs):
        super(AttentionStep, self).__init__(**kwargs)
        self.units = units
        self.dropout = dropout
        self.supports_masking = True
    def build(self, input_shape):
        # Used purely for shape validation.
        super(AttentionStep, self).build(input_shape)

    def call(self, input):
        reverse = K.reverse(input, axes=0)
        reverse_h_a = GRU(self.units, return_sequences=True)(reverse)
        # reverse_h_a = Permute((2, 1))(reverse_h_a)
        a_probs = Dense(1, activation='softmax')(reverse_h_a)
        # a_probs = Permute((2, 1), name='attention_vec')(reverse_a)
        alpha = K.reverse(a_probs, axes=0)
        alpha = K.repeat_elements(alpha,self.units, axis=2)
        print('alpha',alpha.shape)
        reverse_h_b = GRU(self.units, return_sequences=True)(reverse)
        reverse_h_b = Dense(self.units, activation='tanh')(reverse_h_b)
        beta = K.reverse(reverse_h_b, axes=0)
        print('beta',beta.shape)
        e = multiply([beta,alpha])
        c = K.sum(multiply([e,input]),axis=1)
        print('result c_t',c.shape)
        return c

    def compute_output_shape(self, input_shape):
        outputs_shape = (input_shape[0],   self.units)
        return outputs_shape

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'units': self.units,
            'dropout': self.dropout,
        }
        base_config = super(AttentionStep, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

