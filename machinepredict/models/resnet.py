import os

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers.merge import Add
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.regularizers import l2

from .common import channel_axis

#Source:https://github.com/fchollet/keras/issues/2608
def zeropad(x):
    y = K.zeros_like(x)
    return K.concatenate([x, y], axis=1)

def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    shape[1] *= 2
    return tuple(shape)

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Conv2D(
            nb_filter, 
            (nb_row, nb_col), 
            strides=subsample,
            kernel_initializer="he_normal", 
            padding="same")(input)
        norm = BatchNormalization(axis=channel_axis)(conv)
        return Activation("relu")(norm)

    return f

def _bn_relu(x):
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation("relu")(x)

# Helper to build a conv -> BN 
def _conv_bn(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Conv2D(
            nb_filter,
            (nb_row, nb_col),
            strides=subsample,
            kernel_initializer="he_normal", 
            padding="same")(input)
        norm = BatchNormalization(axis=channel_axis)(conv)
        return norm

    return f

# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(axis=channel_axis)(input)
        activation = Activation("relu")(norm)
        return Conv2D(
            nb_filter,
            (nb_row, nb_col),
            strides=subsample,
            kernel_initializer="he_normal", 
            padding="same")(activation)
    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1), option='B'):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual, option=option)

    return f

# conv-bn-relu-conv-bn-relu-sum-bn-relu
#like Lasagne : https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
def _basic_block(nb_filters, init_subsample=(1, 1), option='B'):
    def f(input):
        conv1 = _conv_bn_relu(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _conv_bn_relu(nb_filters, 3, 3)(conv1)
        shortcut = _shortcut(input, residual, option=option)
        shortcut = _bn_relu(shortcut)
        return shortcut
    return f

#conv-bn-relu-conv-bn-sum-relu
# Like original paper reference 
# https://raw.githubusercontent.com/torch/torch.github.io/master/blog/_posts/images/resnets_modelvariants.png
def _ref_block(nb_filters, init_subsample=(1, 1), option='B'):
    def f(input):
        conv1 = _conv_bn_relu(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _conv_bn(nb_filters, 3, 3)(conv1)
        shortcut = _shortcut(input, residual, option=option)
        shortcut = Activation('relu')(shortcut)
        return shortcut
    return f

#conv-bn-relu-conv-bn-sum
# Variant propopsed by the blog
def _refnorelu_block(nb_filters, init_subsample=(1, 1), option='B'):
    def f(input):
        conv1 = _conv_bn_relu(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _conv_bn(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual, option=option)
    return f

# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual, option='B'):
    assert option in ('A', 'B')
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] // residual._keras_shape[2]
    stride_height = input._keras_shape[3] // residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        if option == 'A':
            shortcut = AveragePooling2D((2, 2))(input)
            shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)
        elif option == 'B':
            shortcut = Conv2D(
                residual._keras_shape[1],
                (1, 1),
                strides=(stride_width, stride_height),
                use_bias=False,
                kernel_initializer="he_normal", 
                padding="valid")(input)
    shortcut =  Add()([shortcut, residual])
    return shortcut

# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False, option='B'):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample, option=option)(input)
        return input

    return f

def resnet(params, input_shape , output_shape):
    assert len(output_shape) == 1, "A predictor output should be a vector"
    size_blocks = params['size_blocks']
    nb_filters = params['nb_filters']
    option = params['option']
    block_type = {'bottleneck': _bottleneck, 'basic': _basic_block, "reference": _ref_block, "reference_norelu": _refnorelu_block}
    block_fn = block_type[params['block']]
    output_activation = params['output_activation']
    assert len(size_blocks) == len(nb_filters)
    nb_blocks = len(size_blocks)

    inp = Input(input_shape)
    x = inp
    x = _conv_bn_relu(nb_filter=nb_filters[0], nb_row=3, nb_col=3)(x)
    
    # Build residual blocks..
    for i in range(nb_blocks):
        x = _residual_block(block_fn, nb_filters=nb_filters[i], repetations=size_blocks[i], is_first_layer=(i==0), option=option)(x)
    # Classifier block
    x = GlobalAveragePooling2D()(x)
    out = Dense(output_shape[0], kernel_initializer="he_normal", activation=output_activation)(x)
    return Model(inputs=inp, outputs=out)
