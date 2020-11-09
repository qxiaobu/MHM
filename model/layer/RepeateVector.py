from keras.engine import Layer
from keras import backend as K
class RepeateVector(Layer):

    def __init__(self,n,  **kwargs):
        self.supports_masking = True
        self.n = n
        super(RepeateVector, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],input_shape[2]*self.n)

    def call(self, inputs):
        return K.repeat_elements(inputs, self.n,axis=2)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {'n': self.n}
        base_config = super(RepeateVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))