# -*- coding: utf-8 -*-

# Author: Zhen Zhang <13161411563@163.com>

from keras.engine import Layer
from keras import backend as K
class lastGRU(Layer):

    def __init__( self, **kwargs):
        self.supports_masking = True

        super(lastGRU, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

    def call(self, inputs):
        return inputs[:,-1]

    def compute_mask(self, inputs, mask=None):
        return None

    # def get_config(self):
    #     config = {'n': self.n}
    #     base_config = super(sumVector, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))