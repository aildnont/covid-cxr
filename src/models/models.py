from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant
from tensorflow.keras.applications import MobileNet

def mobilenet_model(model_config, input_shape, metrics, output_bias=None):
    '''
    Define a neural network model with a pre-trained VGG16 base
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
    model = Sequential(name='covid-19-cxr-classifier')
    model.add(MobileNet(include_top=False, input_shape=input_shape))    # MobileNet architecture pre-trained on ImageNet
    model.add(Flatten(name='flatten'))
    model.add(Dense(nodes_dense0, name='dense0'))
    model.add(LeakyReLU())
    model.add(Dropout(dropout, name='dropout0'))
    model.add(Dense(1, activation='sigmoid', name='output'))

    # Set model loss function, optimizer, metrics.
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    model.summary()
    return model

