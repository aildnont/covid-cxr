from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, MaxPool2D, Conv2D, Flatten, LeakyReLU, BatchNormalization, \
    Activation, concatenate, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2
from tensorflow.keras.utils import multi_gpu_model


def dcnn_resnet(model_config, input_shape, metrics, n_classes=2, output_bias=None, gpus=1):
    '''
    Defines a deep convolutional neural network model for multiclass X-ray classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    if model_config['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif model_config['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)  # For now, Adam is default option
    init_filters = model_config['INIT_FILTERS']
    filter_exp_base = model_config['FILTER_EXP_BASE']
    conv_blocks = model_config['CONV_BLOCKS']
    kernel_size = eval(model_config['KERNEL_SIZE'])
    max_pool_size = eval(model_config['MAXPOOL_SIZE'])
    strides = eval(model_config['STRIDES'])

    # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)
    print("MODEL CONFIG: ", model_config)

    # Input layer
    X_input = Input(input_shape)
    X = X_input

    # Add convolutional (residual) blocks
    for i in range(conv_blocks):
        X_res = X
        X = Conv2D(init_filters * (filter_exp_base ** i), kernel_size, strides=strides, padding='same',
                         kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda),
                         name='conv' + str(i) + '_0')(X)
        X = BatchNormalization()(X)
        X = LeakyReLU()(X)
        X = Conv2D(init_filters * (filter_exp_base ** i), kernel_size, strides=strides, padding='same',
                         kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda),
                         name='conv' + str(i) + '_1')(X)
        X = concatenate([X, X_res], name='concat' + str(i))
        X = BatchNormalization()(X)
        X = LeakyReLU()(X)
        X = MaxPool2D(max_pool_size, padding='same')(X)

    # Add fully connected layers
    X = Flatten()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda))(X)
    X = LeakyReLU()(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


def resnet50v2(model_config, input_shape, metrics, n_classes=2, output_bias=None, gpus=1):
    '''
    Defines a model based on a pretrained ResNet50V2 for multiclass X-ray classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    if model_config['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif model_config['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)  # For now, Adam is default option

    # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)
    print("MODEL CONFIG: ", model_config)

    # Start with pretrained ResNet50V2
    X_input = Input(input_shape, name='input_img')
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    X = base_model.output

    # Add custom top
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda))(X)
    X = LeakyReLU()(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model


def resnet101v2(model_config, input_shape, metrics, n_classes=2, output_bias=None, gpus=1):
    '''
    Defines a model based on a pretrained ResNet101V2 for multiclass X-ray classification.
    Note that batch size per GPU should be >= 12 to prevent NaN in batch normalization.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :return: a Keras Model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    if model_config['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif model_config['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)  # For now, Adam is default option

    # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)
    print("MODEL CONFIG: ", model_config)

    # Start with pretrained ResNet101V2
    X_input = Input(input_shape, name='input_img')
    base_model = ResNet101V2(include_top=False, weights='imagenet', input_shape=input_shape, input_tensor=X_input)
    X = base_model.output

    # Add custom top
    X = GlobalAveragePooling2D()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda))(X)
    X = LeakyReLU()(X)
    X = Dense(n_classes, bias_initializer=output_bias)(X)
    Y = Activation('softmax', dtype='float32', name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.summary()
    if gpus >= 2:
        model = multi_gpu_model(model, gpus=gpus)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    return model