import pandas as pd
import yaml
import os
import tensorflow as tf
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from math import ceil
import datetime
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.models.models import *
from src.visualization.visualize import *
from src.custom.metrics import F1Score
from src.data.preprocess import remove_text

def get_class_weights(histogram, class_multiplier=None):
    '''
    Computes weights for each class to be applied in the loss function during training.
    :param histogram: A list depicting the number of each item in different class
    :param class_multiplier: List of values to multiply the calculated class weights by. For further control of class weighting.
    :return: A dictionary containing weights for each class
    '''
    weights = [None] * len(histogram)
    for i in range(len(histogram)):
        weights[i] = (1.0 / len(histogram)) * sum(histogram) / histogram[i]
    class_weight = {i: weights[i] for i in range(len(histogram))}
    if class_multiplier is not None:
        class_weight = [class_weight[i] * class_multiplier[i] for i in range(len(histogram))]
    class_weight[0] *= 3
    print("Class weights: ", class_weight)
    return class_weight


def random_minority_oversample(train_set):
    '''
    Oversample the minority class using the specified algorithm
    :param train_set: Training set image file names and labels
    :return: A new training set containing oversampled examples
    '''
    X_train = train_set['filename'].to_numpy()
    Y_train = train_set['label'].to_numpy()
    sampler = RandomOverSampler(random_state=np.random.randint(0, high=1000))
    X_resampled, Y_resampled = sampler.fit_resample(X_train, Y_train)
    print("Train set shape before oversampling: ", X_train.shape, " Train set shape after resampling: ", X_resampled.shape)
    train_set_resampled = pd.DataFrame({'filename': X_resampled, 'label':Y_resampled})
    return train_set_resampled


def train_model(cfg, data, model, callbacks, verbose=1):
    '''
    Train a and evaluate model on given data.
    :param cfg: Project config (from config.yml)
    :param data: dict of partitioned dataset
    :param model: Keras model to train
    :param callbacks: list of callbacks for Keras model
    :param verbose: Verbosity mode to pass to model.fit_generator()
    :return: Trained model and associated performance metrics on the test set
    '''

    # Apply class imbalance strategy. We have many more X-rays negative for COVID-19 than positive.
    histogram = np.bincount(data['TRAIN']['label'].astype(int))
    class_weight = None
    class_multiplier = None
    if cfg['TRAIN']['CLASS_MODE'] == 'multiclass':
        class_multiplier = cfg['TRAIN']['CLASS_MULTIPLIER']
    if cfg['TRAIN']['IMB_STRATEGY'] == 'class_weight':
        class_weight = get_class_weights(histogram, class_multiplier)
    else:
        data['TRAIN'] = random_minority_oversample(data['TRAIN'])

    # Create ImageDataGenerators
    train_img_gen = ImageDataGenerator(rescale=1.0/255.0, preprocessing_function=remove_text)
    val_img_gen = ImageDataGenerator(rescale=1.0/255.0, preprocessing_function=remove_text)
    test_img_gen = ImageDataGenerator(rescale=1.0/255.0, preprocessing_function=remove_text)

    # Create DataFrameIterators
    img_shape = tuple(cfg['DATA']['IMG_DIM'])
    if cfg['TRAIN']['CLASS_MODE'] == 'binary':
        y_col = 'label'
        class_mode = 'raw'
    else:
        y_col = 'label_str'
        class_mode = 'categorical'
    train_generator = train_img_gen.flow_from_dataframe(dataframe=data['TRAIN'], directory=cfg['PATHS']['TRAIN_IMGS'],
        x_col="filename", y_col=y_col, target_size=img_shape, batch_size=cfg['TRAIN']['BATCH_SIZE'], class_mode=class_mode)
    val_generator = val_img_gen.flow_from_dataframe(dataframe=data['VAL'], directory=cfg['PATHS']['VAL_IMGS'],
        x_col="filename", y_col=y_col, target_size=img_shape, batch_size=cfg['TRAIN']['BATCH_SIZE'], class_mode=class_mode)
    test_generator = test_img_gen.flow_from_dataframe(dataframe=data['TEST'], directory=cfg['PATHS']['TEST_IMGS'],
        x_col="filename", y_col=y_col, target_size=img_shape, batch_size=cfg['TRAIN']['BATCH_SIZE'], class_mode=class_mode,
        shuffle=False)

    # Train the model.
    steps_per_epoch = ceil(train_generator.n / train_generator.batch_size)
    val_steps = ceil(val_generator.n / val_generator.batch_size)
    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=cfg['TRAIN']['EPOCHS'],
                                  validation_data=val_generator, validation_steps=val_steps, callbacks=callbacks,
                                  verbose=verbose, class_weight=class_weight)

    # Run the model on the test set and print the resulting performance metrics.
    test_results = model.evaluate_generator(test_generator, verbose=1)
    test_metrics = {}
    test_summary_str = [['**Metric**', '**Value**']]
    for metric, value in zip(model.metrics_names, test_results):
        test_metrics[metric] = value
        print(metric, ' = ', value)
        test_summary_str.append([metric, str(value)])
    return model, test_metrics, test_generator


def train_experiment(experiment='single_train', save_weights=True, write_logs=True):
    '''
    Defines and trains HIFIS-v2 model. Prints and logs relevant metrics.
    :param experiment: The type of training experiment. Choices are {'single_train'}
    :param save_weights: A flag indicating whether to save the model weights
    :param write_logs: A flag indicating whether to write TensorBoard logs
    :return: A dictionary of metrics on the test set
    '''

    # Load project config data
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # Load dataset file paths and labels
    data = {}
    data['TRAIN'] = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
    data['VAL'] = pd.read_csv(cfg['PATHS']['VAL_SET'])
    data['TEST'] = pd.read_csv(cfg['PATHS']['TEST_SET'])

    # Define metrics.
    thresholds = cfg['TRAIN']['THRESHOLDS']     # Load classification thresholds
    if cfg['TRAIN']['CLASS_MODE'] == 'binary':
        covid_class_id = None
    else:
        covid_class_id = 0
    metrics = ['accuracy', BinaryAccuracy(name='accuracy'),
               Precision(name='precision', thresholds=thresholds, class_id=covid_class_id),
               Recall(name='recall', thresholds=thresholds, class_id=covid_class_id),
               AUC(name='auc'),
               F1Score(name='f1score', thresholds=thresholds, class_id=covid_class_id)]

    # Set callbacks.
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
    callbacks = [early_stopping]
    if write_logs:
        log_dir = cfg['PATHS']['LOGS'] + "training\\" + cur_date
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)

    # Define the model.
    histogram = list(np.bincount(data['TRAIN']['label'].astype(int)))
    print(['Class ' + str(i) + ': ' + str(histogram[i]) + '. ' for i in range(len(histogram))])
    input_shape = cfg['DATA']['IMG_DIM'] + [3]
    if cfg['TRAIN']['CLASS_MODE'] == 'binary':
        output_bias = np.log([histogram[1] / histogram[0]])
        model = dcnn_binary(cfg['NN']['DCNN_BINARY'], input_shape, metrics, output_bias=output_bias)
    else:
        n_classes = len(cfg['DATA']['CLASSES'])
        model = dcnn_multiclass(cfg['NN']['DCNN_MULTICLASS'], input_shape, n_classes, metrics)

    # Conduct desired train experiment
    if experiment == 'single_train':
        model, test_metrics, test_generator = train_model(cfg, data, model, callbacks)

    # Visualization of test results
    test_predictions = model.predict_generator(test_generator, verbose=0)
    roc_img = plot_roc("Test set", data['TEST']['label'], test_predictions, dir_path=None)
    cm_img = plot_confusion_matrix(data['TEST']['label'], test_predictions, dir_path=None)

    # Log test set results and plots in TensorBoard
    if write_logs:
        writer = tf.summary.create_file_writer(logdir=log_dir)
        test_summary_str = [['**Metric**','**Value**']]
        for metric in test_metrics:
            if metric in ['precision', 'recall'] and isinstance(metric, list):
                metric_values = dict(zip(thresholds, test_metrics[metric]))
            else:
                metric_values = test_metrics[metric]
            test_summary_str.append([metric, str(metric_values)])
        with writer.as_default():
            tf.summary.text(name='Test set metrics', data=tf.convert_to_tensor(test_summary_str), step=0)
            tf.summary.image(name='ROC Curve (Test Set)', data=roc_img, step=0)
            tf.summary.image(name='Confusion Matrix (Test Set)', data=cm_img, step=0)

    # Save the model's weights
    if save_weights:
        model_path = os.path.splitext(cfg['PATHS']['MODEL_WEIGHTS'])[0] + cur_date + '.h5'
        save_model(model, model_path)
    return test_metrics


if __name__ == '__main__':
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    results = train_experiment(experiment=cfg['TRAIN']['EXPERIMENT_TYPE'], save_weights=True, write_logs=True)