from keras.engine import Layer
from keras import backend as K
class Reverse(Layer):

    def __init__(self,  **kwargs):
        self.supports_masking = True
        super(Reverse, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],input_shape[2])

    def call(self, inputs):
        return K.reverse(inputs, axes=0)

    def compute_mask(self, inputs, mask=None):
        return None

    # def get_config(self):
    #     config = {'n': self.n}
    #     base_config = super(Reverse, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))