from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, MaxPool2D, Conv2D, Flatten, LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant


def dcnn_binary(model_config, input_shape, metrics, output_bias=None):
    '''
    Defines a deep convolutional neural network model for binary X-ray classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :param output_bias: initial bias applied to output layer
    :return: a Keras Sequential model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    init_filters = model_config['INIT_FILTERS']
    filter_exp_base = model_config['FILTER_EXP_BASE']
    conv_blocks = model_config['CONV_BLOCKS']
    kernel_size = eval(model_config['KERNEL_SIZE'])
    max_pool_size = eval(model_config['MAXPOOL_SIZE'])
    strides = eval(model_config['STRIDES'])
    if output_bias is not None:
        output_bias = Constant(output_bias)

    model = Sequential(name='covid-19-cxr-custom1')

    # Add convolutional blocks
    for i in range(conv_blocks):
        if i == 0:
            model.add(Conv2D(init_filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_uniform',
                             activity_regularizer=l2(l2_lambda), input_shape=input_shape))
        else:
            model.add(Conv2D(init_filters * (filter_exp_base ** i), kernel_size, strides=strides, padding='same',
                             kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPool2D(max_pool_size, padding='same'))

    # Add fully connected layers
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda)))
    model.add(LeakyReLU())
    model.add(Dense(1, activation='sigmoid', bias_initializer=output_bias, name='output'))

    # Set model loss function, optimizer, metrics.
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()
    return model


def dcnn_multiclass(model_config, input_shape, num_classes, metrics):
    '''
    Defines a deep convolutional neural network model for multiclass X-ray classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param num_classes: Number of output classes
    :param metrics: Metrics to track model's performance
    :return: a Keras Sequential model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    optimizer = Adam(learning_rate=lr)
    init_filters = model_config['INIT_FILTERS']
    filter_exp_base = model_config['FILTER_EXP_BASE']
    conv_blocks = model_config['CONV_BLOCKS']
    kernel_size = eval(model_config['KERNEL_SIZE'])
    max_pool_size = eval(model_config['MAXPOOL_SIZE'])
    strides = eval(model_config['STRIDES'])

    model = Sequential(name='covid-19-cxr-custom1')

    # Add convolutional blocks
    for i in range(conv_blocks):
        if i == 0:
            model.add(Conv2D(init_filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_uniform',
                             activity_regularizer=l2(l2_lambda), input_shape=input_shape))
        else:
            model.add(Conv2D(init_filters * (filter_exp_base ** i), kernel_size, strides=strides, padding='same',
                             kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPool2D(max_pool_size, padding='same'))

    # Add fully connected layers
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda)))
    model.add(LeakyReLU())
    model.add(Dense(num_classes, activation='softmax', name='output'))

    # Set model loss function, optimizer, metrics.
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()
    return model