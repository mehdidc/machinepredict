from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import ZeroPadding2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.models import Model
from .common import act
from .common import channel_axis

def lenet(params, input_shape, output_shape):
    assert len(output_shape) == 1, "A predictor output should be a vector"
 
    nb_filters = params['nb_filters']
    pr = params['dropout']
    fc_pr = params['fc_dropout']
    batch_norm = params['batch_norm']
    fc = params['fc']
    size_filters = params['size_filters']
    activation = params['activation']
    output_activation = params['output_activation']
    init = 'glorot_uniform'
    inp = Input(input_shape)
    x = inp
    x = ZeroPadding2D(padding=(1, 1))(x)
    for k in nb_filters:
        x = Conv2D(k, (size_filters, size_filters), kernel_initializer=init)(x)
        if batch_norm:
            x = BatchNormalization(axis=channel_axis)(x)
        x = act(x, activation=activation)
        x = MaxPooling2D((2, 2))(x)
        if pr > 0:
            x = Dropout(pr)(x)
    x = Flatten()(x)
    for units in fc:
        x = Dense(units, kernel_initializer=init)(x)
        x = act(x, activation=activation)
        x = Dropout(fc_pr)(x)
    x = Dense(output_shape[0], kernel_initializer=init)(x)
    out = act(x, activation=output_activation)
    return Model(inputs=inp, outputs=out)
