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

class HierarchicalAttention(Layer):
    """ Return the outputs and last_output
    """
    def __init__(self,alpha1,alpha2,alpha3, **kwargs):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        super(HierarchicalAttention, self).__init__(**kwargs)
        self.supports_masking = True
    def build(self, input_shape):
        # Used purely for shape validation.
        super(HierarchicalAttention, self).build(input_shape)

    def call(self, input):
        sum = Add()([self.alpha1,self.alpha2,self.alpha3])
        alpha1 = tf.div(self.alpha1,sum)
        alpha2 = tf.div(self.alpha2,sum)
        alpha3 = tf.div(self.alpha3,sum)
        GLoss = Add()([tf.multiply(alpha1, input[0]), tf.multiply(alpha2, input[1]), tf.multiply(alpha3, input[2])])
        # GLoss = Dense(self.units,activation='softmax')(GLoss)
        return [GLoss,alpha1,alpha2,alpha3]
    def compute_output_shape(self, input_shape):
        return [input_shape[0],1,1,1]

    def compute_mask(self, inputs, mask=None):
        return None

