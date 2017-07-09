from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.regularizers import l2

from .common import channel_axis


def conv_factory(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, 
                (3, 3),
                kernel_initializer="he_uniform",
                padding="same",
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


def transition(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """

    x = BatchNormalization(axis=channel_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, 
               (1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x


def denseblock(x, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=channel_axis)(list_feat)
        nb_filter += growth_rate

    return x, nb_filter


def denseblock_altern(x, nb_layers, nb_filter, growth_rate,
                      dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_factory
       is fed to subsequent ones. (Alternative of a above)

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    * The main difference between this implementation and the implementation
    above is that the one above
    """
    for i in range(nb_layers):
        merge_tensor = conv_factory(x, growth_rate, dropout_rate, weight_decay)
        x = merge([merge_tensor, x], mode='concat', concat_axis=channel_axis)
        nb_filter += growth_rate
    return x, nb_filter


def densenet(params, input_shape, output_shape):
    assert len(output_shape) == 1, "A predictor output should be a vector"
    depth = params['depth']
    nb_dense_block = params['nb_dense_block']
    growth_rate = params['growth_rate']
    nb_filter = params['nb_filter']
    output_activation = params['output_activation']
    dropout_rate = params['dropout_rate']
    weight_decay = params['weight_decay']
    model = _densenet(
        input_shape, 
        output_shape[0], 
        depth, 
        nb_dense_block, 
        growth_rate, 
        nb_filter,
        output_activation,
        dropout_rate,
        weight_decay)
    return model


def _densenet(input_shape, nb_outputs, depth, nb_dense_block, growth_rate,
              nb_filter, output_activation, dropout_rate=None, weight_decay=1E-4):
    """ Build the DenseNet model
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    model_input = Input(input_shape)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(
        nb_filter, 
        (3, 3),
        kernel_initializer="he_uniform",
        padding="same",
        name="initial_conv2D",
        use_bias=False,
        kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x = transition(x, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, nb_layers, nb_filter, growth_rate,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x = BatchNormalization(
        axis=channel_axis,
        gamma_regularizer=l2(weight_decay),
        beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_outputs,
              activation=output_activation,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)
    densenet = Model(inputs=model_input, outputs=x, name="DenseNet")
    return densenet
