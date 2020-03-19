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

def get_class_weights(num_pos, num_neg, pos_weight=0.5):
    '''
    Computes weights for each class to be applied in the loss function during training.
    :param num_pos: # positive samples
    :param num_neg: # negative samples
    :param pos_weight: The relative amount to further weigh the positive class
    :return: A dictionary containing weights for each class
    '''
    weight_neg = (1 - pos_weight) * (num_neg + num_pos) / (num_neg)
    weight_pos = pos_weight * (num_neg + num_pos) / (num_pos)
    class_weight = {0: weight_neg, 1: weight_pos}
    print("Class weights: Class 0 = {:.2f}, Class 1 = {:.2f}".format(weight_neg, weight_pos))
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

    # Apply class imbalance strategy
    num_neg, num_pos = np.bincount(data['TRAIN']['label'].astype(int))
    class_weight = None
    if cfg['TRAIN']['IMB_STRATEGY'] == 'class_weight':
        class_weight = get_class_weights(num_pos, num_neg, cfg['TRAIN']['POS_WEIGHT'])
    else:
        data['TRAIN'] = random_minority_oversample(data['TRAIN'])

    # Create ImageDataGenerators
    train_img_gen = ImageDataGenerator(rescale=1.0/255.0)
    val_img_gen = ImageDataGenerator(rescale=1.0/255.0)
    test_img_gen = ImageDataGenerator(rescale=1.0/255.0)

    # Create DataFrameIterators
    img_shape = tuple(cfg['DATA']['IMG_DIM'])
    train_generator = train_img_gen.flow_from_dataframe(dataframe=data['TRAIN'], directory=cfg['PATHS']['TRAIN_IMGS'],
        x_col="filename", y_col="label", target_size=img_shape, batch_size=cfg['TRAIN']['BATCH_SIZE'], class_mode='raw')
    val_generator = val_img_gen.flow_from_dataframe(dataframe=data['VAL'], directory=cfg['PATHS']['VAL_IMGS'],
        x_col="filename", y_col="label", target_size=img_shape, batch_size=cfg['TRAIN']['BATCH_SIZE'], class_mode='raw')
    test_generator = test_img_gen.flow_from_dataframe(dataframe=data['TEST'], directory=cfg['PATHS']['TEST_IMGS'],
        x_col="filename", y_col="label", target_size=img_shape, batch_size=cfg['TRAIN']['BATCH_SIZE'], class_mode='raw')

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
    metrics = ['accuracy', BinaryAccuracy(name='accuracy'), Precision(name='precision', thresholds=thresholds),
               Recall(name='recall', thresholds=thresholds), AUC(name='auc')]

    # Set callbacks.
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
    callbacks = [early_stopping]
    if write_logs:
        log_dir = cfg['PATHS']['LOGS'] + "training\\" + cur_date
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)

    # Define the model.
    num_neg, num_pos = np.bincount(data['TRAIN']['label'].astype(int))
    print("Class 0:", num_neg, "XRs. Class 1:", num_pos, "XRs.")
    output_bias = np.log([num_pos / num_neg])
    input_shape = cfg['DATA']['IMG_DIM'] + [3]
    model = dcnn1(cfg['NN']['DCNN1'], input_shape, metrics, output_bias=output_bias)   # Build model graph

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

    if save_weights:
        model_path = os.path.splitext(cfg['PATHS']['MODEL_WEIGHTS'])[0] + cur_date + '.h5'
        save_model(model, model_path)        # Save model weights
    return test_metrics


if __name__ == '__main__':
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    results = train_experiment(experiment=cfg['TRAIN']['EXPERIMENT_TYPE'], save_weights=True, write_logs=True)