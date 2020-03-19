import numpy as np
import pandas as pd
import yaml
import os
import glob
import pathlib
import shutil
from math import ceil
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def preprocess():
    '''
    Preprocess and partition image data. Assemble all image file paths and partition into training, validation and
    test sets. Copy raw images to folders for training, validation and test sets.
    '''

    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))  # Load config data
    covid_data_path = cfg['PATHS']['RAW_COVID_DATA']
    other_data_path = cfg['PATHS']['RAW_OTHER_DATA']
    processed_path = cfg['PATHS']['PROCESSED_DATA']

    # Assemble filenames comprising COVID dataset
    metadata_file_path = covid_data_path + 'metadata.csv'
    covid_df = pd.read_csv(metadata_file_path)
    PA_cxrs_df = (covid_df['view'] == 'PA')
    covid_patients_df = covid_df['finding'] == 'COVID-19'
    PA_covid_df = covid_df[covid_patients_df & PA_cxrs_df]      # PA images diagnosed COVID
    PA_covid_df['label'] = 1
    PA_other_df = covid_df[~covid_patients_df & PA_cxrs_df]     # PA images with other diagnoses
    PA_other_df['label'] = 0
    file_df = pd.concat([PA_covid_df[['filename', 'label']], PA_other_df[['filename', 'label']]], axis=0)
    file_df['filename'] = covid_data_path + 'images\\' + file_df['filename'].astype(str)    # Set as absolute paths

    # Assemble filenames comprising Kaggle dataset that is organized into "normal" and "pneumonia" XRs
    normal_xr_filenames = [(other_data_path + 'normal\\' + f) for f in os.listdir(other_data_path + 'normal\\') if
                  os.path.isfile(os.path.join(other_data_path + 'normal\\', f))]
    normal_xr_filenames = normal_xr_filenames[0: ceil(0.5 * len(normal_xr_filenames))]
    pneum_xr_filenames = [(other_data_path + 'pneumonia\\' + f) for f in os.listdir(other_data_path + 'pneumonia\\') if
                       os.path.isfile(os.path.join(other_data_path + 'pneumonia\\', f))]
    pneum_xr_filenames = pneum_xr_filenames[0: ceil(0.5 * len(pneum_xr_filenames))]
    other_file_df = pd.DataFrame({'filename': normal_xr_filenames + pneum_xr_filenames, 'label': 0})

    # Combine both datasets
    file_df = pd.concat([file_df, other_file_df], axis=0)

    # Split dataset into train, val and test sets
    val_split = cfg['DATA']['VAL_SPLIT']
    test_split = cfg['DATA']['TEST_SPLIT']
    file_df_train, file_df_test = train_test_split(file_df, test_size=test_split, stratify=file_df['label'])
    relative_val_split = val_split / (1 - test_split)  # Calculate fraction of train_df to be used for validation
    file_df_train, file_df_val = train_test_split(file_df_train, test_size=relative_val_split,
                                                      stratify=file_df_train['label'])

    # Delete old datasets
    dest_dir = os.path.join(os.getcwd(), cfg['PATHS']['PROCESSED_DATA'])
    print('Deleting old sets.')
    for path in pathlib.Path(os.path.join(dest_dir, 'train\\')).glob('*'):
        if '.gitkeep' not in str(path):
            path.unlink()
    for path in pathlib.Path(os.path.join(dest_dir, 'val\\')).glob('*'):
        if '.gitkeep' not in str(path):
            path.unlink()
    for path in pathlib.Path(os.path.join(dest_dir, 'test\\')).glob('*'):
        if '.gitkeep' not in str(path):
            path.unlink()

    # Copy images to appropriate directories
    print('Copying training set images.')
    for file_path in tqdm(file_df_train['filename'].tolist()):
        shutil.copy(file_path, os.path.join(dest_dir, 'train\\'))
    print('Copying validation set images.')
    for file_path in tqdm(file_df_val['filename'].tolist()):
        shutil.copy(file_path, os.path.join(dest_dir, 'val\\'))
    print('Copying test set images.')
    for file_path in tqdm(file_df_test['filename'].tolist()):
        shutil.copy(file_path, os.path.join(dest_dir, 'test\\'))

    # Update file path dataframes
    file_df_train['filename'] = file_df_train['filename'].str.split('\\').str[-1]
    file_df_val['filename'] = file_df_val['filename'].str.split('\\').str[-1]
    file_df_test['filename'] = file_df_test['filename'].str.split('\\').str[-1]

    # Save training, validation and test sets
    file_df_train.to_csv(cfg['PATHS']['TRAIN_SET'])
    file_df_val.to_csv(cfg['PATHS']['VAL_SET'])
    file_df_test.to_csv(cfg['PATHS']['TEST_SET'])
    return

if __name__ == '__main__':
    preprocess()