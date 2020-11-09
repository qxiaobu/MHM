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

class Hierarchica(Layer):
    """ Return the outputs and last_output
    """
    def __init__(self, units,inputdim, dropout=0., **kwargs):
        super(Hierarchica, self).__init__(**kwargs)
        self.units = units
        self.inputdim = inputdim
        self.dropout = dropout
        self.supports_masking = True
    def build(self, input_shape):
        # Used purely for shape validation.
        self.kernel = self.add_weight(name='kernel',
                                              shape=(self.units,1),
                                              initializer='uniform',
                                              trainable=True)
        self.bias = self.add_weight(shape=(1,),
                                            initializer='uniform',
                                            name='bias',
                                            trainable=True)
        super(Hierarchica, self).build(input_shape)

    def call(self, input):
        AG1 = Dense(self.inputdim,activation='relu')(input)
        AG2 = Dense(self.inputdim,activation='relu')(multiply([AG1,input]))
        PG = Dense(self.units, activation='sigmoid')(AG2)
        AL1 = Dense(self.inputdim,activation='relu')(AG1)
        PL1 = Dense(self.units,activation='sigmoid')(AL1)
        AL2 = Dense(self.inputdim,activation='relu')(AG2)
        PL2 = Dense(self.units,activation='sigmoid')(AL2)
        #
        alpha1 = K.dot(PL1,self.kernel)
        alpha1 = K.bias_add(alpha1, self.bias, data_format='channels_last')
        alpha2 = K.dot(PL2,self.kernel)
        alpha2 = K.bias_add(alpha2, self.bias, data_format='channels_last')
        alpha3 = K.dot(PG,self.kernel)
        alpha3 = K.bias_add(alpha3, self.bias, data_format='channels_last')
        sum = alpha1+alpha2+alpha3
        alpha1 = alpha1/sum
        alpha2 = alpha2/sum
        alpha3 = alpha3/sum
        GLoss = Add()([tf.multiply(alpha1,PL1),tf.multiply(alpha2,PL2),tf.multiply(alpha3,PG)])
        # GLoss = Dense(self.units,activation='softmax')(GLoss)
        return GLoss
    def compute_output_shape(self, input_shape):
        global_shape = (input_shape[0], self.units)
        return global_shape

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'units': self.units,
            'dropout': self.dropout,
        }
        base_config = super(Hierarchica, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

