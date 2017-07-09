from collections import namedtuple
from keras.layers import PReLU

def act(x, activation='relu'):
    if activation == 'prelu':
        x = PReLU()(x)
    else:
        x = Activation(activation)(x)
    return x
