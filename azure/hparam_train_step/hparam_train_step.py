import os
import argparse
import yaml
import datetime
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, Callback
from azureml.core import Run
from src.train import train_model
from src.visualization.visualize import plot_roc, plot_confusion_matrix


# Receive input arguments from pipeline call, including preprocessed data directory and values of hyperparameters.
parser = argparse.ArgumentParser()
parser.add_argument('--rawdatadir', type=str, help="root directory for all datasets")
parser.add_argument('--preprocesseddir', type=str, help="preprocessed data directory")
parser.add_argument('--KERNEL_SIZE', type=str)
parser.add_argument('--MAXPOOL_SIZE', type=str)
parser.add_argument('--INIT_FILTERS', type=int)
parser.add_argument('--FILTER_EXP_BASE', type=int)
parser.add_argument('--CONV_BLOCKS', type=int)
parser.add_argument('--NODES_DENSE0', type=int)
parser.add_argument('--LR', type=float)
parser.add_argument('--OPTIMIZER', type=str)
parser.add_argument('--DROPOUT', type=float)
parser.add_argument('--L2_LAMBDA', type=float)
args = parser.parse_args()

# Get reference to current run
run = Run.get_context()

# Update paths of input data in config to represent paths on blob.
cfg = yaml.full_load(open(os.getcwd() + "./config.yml", 'r'))  # Load config data
cfg['PATHS']['RAW_DATA'] = args.rawdatadir
cfg['PATHS']['PROCESSED_DATA'] = args.preprocesseddir
cfg['PATHS']['TRAIN_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['TRAIN_SET'].split('/')[-1]
cfg['PATHS']['VAL_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['VAL_SET'].split('/')[-1]
cfg['PATHS']['TEST_SET'] = cfg['PATHS']['PROCESSED_DATA'] + '/' + cfg['PATHS']['TEST_SET'].split('/')[-1]

# Set hyperparameter values in project config based on values supplied in args list.
args_dict = vars(args)
model_type = 'DCNN_BINARY' if cfg['TRAIN']['CLASS_MODE'] == 'binary' else 'DCNN_MULTICLASS'
for arg in args_dict:
    if arg in cfg['TRAIN']:
        cfg['TRAIN'][arg] = args_dict[arg]
    if arg in cfg['NN'][model_type]:
        cfg['NN'][model_type][arg] = args_dict[arg]

# Set logs directory according to datetime
cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# Load dataset file paths and labels
data = {}
data['TRAIN'] = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
data['VAL'] = pd.read_csv(cfg['PATHS']['VAL_SET'])
data['TEST'] = pd.read_csv(cfg['PATHS']['TEST_SET'])

# Custom Keras callback that logs all training and validation metrics after each epoch to the current Azure run
class LogRunMetrics(Callback):
    def on_epoch_end(self, epoch, log):
        for metric_name in log:
            if 'val' in metric_name:
                run.log('validation_' + metric_name.split('_')[-1], log[metric_name])
            else:
                run.log('training_' + metric_name, log[metric_name])
        #run.log('validation_auc', log['val_auc'])

# Set model callbacks
callbacks = [EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min', restore_best_weights=True),
             LogRunMetrics()]

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