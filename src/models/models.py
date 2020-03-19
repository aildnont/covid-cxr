from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, MaxPool2D, Conv2D, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications import MobileNet

def mobilenet_model(model_config, input_shape, metrics, output_bias=None):
    '''
    Define a neural network model with a pre-trained MobileNet base
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param output_bias: initial bias applied to output layer
    :return: a Keras model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES']['DENSE0']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    optimizer = Adam(learning_rate=lr)
    if output_bias is not None:
        output_bias = Constant(output_bias)

    # Define model architecture
    model = Sequential(name='covid-19-cxr-mobilenet')
    model.add(MobileNet(include_top=False, input_shape=input_shape))    # MobileNet architecture pre-trained on ImageNet
    model.add(MaxPool2D((2,2), padding='same', name='maxpool0'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(nodes_dense0, name='dense0'))
    model.add(LeakyReLU())
    model.add(Dropout(dropout, name='dropout0'))
    model.add(Dense(1, activation='sigmoid', bias_initializer=output_bias, name='output'))

    # Set model loss function, optimizer, metrics.
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    model.summary()
    return model


def dcnn1(model_config, input_shape, metrics, output_bias=None):
    '''
    Define a neural network model with a pre-trained MobileNet base
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param output_bias: initial bias applied to output layer
    :return: a Keras model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES']['DENSE0']
    nodes_dense1 = model_config['NODES']['DENSE1']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    init_filters = model_config['INIT_FILTERS']
    filter_exp = model_config['FILTER_EXP']
    kernel_size = eval(model_config['KERNEL_SIZE'])
    strides = eval(model_config['STRIDES'])
    if output_bias is not None:
        output_bias = Constant(output_bias)

    model = Sequential(name='covid-19-cxr-custom1')

    model.add(Conv2D(init_filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_uniform',
                     activity_regularizer=l2(l2_lambda), input_shape=input_shape))
    model.add(Conv2D(init_filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_uniform',
                     activity_regularizer=l2(l2_lambda)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool2D((2,2), padding='same'))

    model.add(Conv2D(init_filters * filter_exp, kernel_size, strides=strides, padding='same',
                     activity_regularizer=l2(l2_lambda), kernel_initializer='he_uniform'))
    model.add(Conv2D(init_filters * filter_exp, kernel_size, strides=strides, padding='same',
                     activity_regularizer=l2(l2_lambda), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool2D((2,2), padding='same'))

    model.add(Conv2D(init_filters * (filter_exp ** 2), kernel_size, strides=strides, padding='same',
                     activity_regularizer=l2(l2_lambda), kernel_initializer='he_uniform'))
    model.add(Conv2D(init_filters * (filter_exp ** 2), kernel_size, strides=strides, padding='same',
                     activity_regularizer=l2(l2_lambda), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool2D((2,2), padding='same'))

    model.add(Conv2D(init_filters * (filter_exp ** 3), kernel_size, strides=strides, padding='same',
                     activity_regularizer=l2(l2_lambda), kernel_initializer='he_uniform'))
    model.add(Conv2D(init_filters * (filter_exp ** 3), kernel_size, strides=strides, padding='same',
                     activity_regularizer=l2(l2_lambda), kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool2D((2,2), padding='same'))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda)))
    model.add(LeakyReLU())
    model.add(Dense(1, activation='sigmoid', bias_initializer=output_bias, name='output'))

    # Set model loss function, optimizer, metrics.
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()
    return model