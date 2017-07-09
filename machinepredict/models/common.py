from keras.layers import Activation
from keras.layers import PReLU
import keras.backend as K

channel_axis = 3 if K.image_data_format() == 'channels_last' else 1


def act(x, activation='relu'):
    if activation == 'prelu':
        x = PReLU()(x)
    else:
        x = Activation(activation)(x)
    return x
