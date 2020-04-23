import os
import argparse
import yaml
import datetime
import pandas as pd
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
import tensorflow as tf
from azureml.core import Run
from src.train import train_model
from src.visualization.visualize import plot_roc, plot_confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--rawdatadir', type=str, help="root directory for all datasets")
parser.add_argument('--preprocesseddir', type=str, help="preprocessed data directory")
parser.add_argument('--traininglogsdir', type=str, help="training logs directory")
parser.add_argument('--modelsdir', type=str, help="models directory")
args = parser.parse_args()
run = Run.get_context()
print("GPUs available:")
print(tf.config.experimental.list_physical_devices('GPU'))

# Update paths of input data in config to represent paths on blob.
cfg = yaml.full_load(open(os.getcwd() + "./config.yml", 'r'))  # Load config data
cfg['PATHS']['RAW_DATA'] = args.rawdatadir
cfg['PATHS']['PROCESSED_DATA'] = args.preprocesseddir
cfg['PATHS']['TRAIN_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['TRAIN_SET'].split('/')[-1]
cfg['PATHS']['VAL_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['VAL_SET'].split('/')[-1]
cfg['PATHS']['TEST_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['TEST_SET'].split('/')[-1]

# Set paths to run's ./output/ directory
cfg['PATHS']['LOGS'] = args.traininglogsdir
cfg['PATHS']['MODEL_WEIGHTS'] = args.modelsdir

# Set logs directory according to datetime
cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = cfg['PATHS']['LOGS'] + cur_date

# Load dataset file paths and labels
data = {}
data['TRAIN'] = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
data['VAL'] = pd.read_csv(cfg['PATHS']['VAL_SET'])
data['TEST'] = pd.read_csv(cfg['PATHS']['TEST_SET'])

# Custom Keras callback that logs all training and validation metrics after each epoch to the current Azure run
class LogRunMetrics(Callback):
    def on_epoch_end(self, epoch, log):
        for metric_name in log:
            run.log(metric_name, log[metric_name])

# Set model callbacks
callbacks = [EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min', restore_best_weights=True),
             LogRunMetrics()]
if cfg['PATHS']['LOGS'] is not None:
    callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))

# Train a model
start_time = datetime.datetime.now()
model, test_metrics, test_generator = train_model(cfg, data, callbacks)
print("TRAINING TIME = " + str((datetime.datetime.now() - start_time).total_seconds() / 60.0) + " min")

# Log test set performance metrics, ROC, confusion matrix in Azure run
test_predictions = model.predict_generator(test_generator, verbose=0)
test_labels = test_generator.labels
for metric_name in test_metrics:
    run.log('test_' + metric_name, test_metrics[metric_name])
covid_idx = test_generator.class_indices['COVID-19']
roc_plt = plot_roc("Test set", test_generator.labels, test_predictions, class_id=covid_idx)
run.log_image("ROC", plot=roc_plt)
cm_plt = plot_confusion_matrix(test_generator.labels, test_predictions, class_id=covid_idx)
run.log_image("Confusion matrix", plot=cm_plt)

# Save the model's weights
if cfg['PATHS']['MODEL_WEIGHTS'] is not None:
    if not os.path.exists(cfg['PATHS']['MODEL_WEIGHTS']):
        os.makedirs(cfg['PATHS']['MODEL_WEIGHTS'])
    save_model(model, cfg['PATHS']['MODEL_WEIGHTS'] + 'model' + cur_date + '.h5')  # Save the model's weights