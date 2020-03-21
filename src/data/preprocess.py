import numpy as np
import pandas as pd
import yaml
import os
import pathlib
import shutil
import cv2
from math import ceil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def build_dataset(cfg, covid_data_path, kaggle_data_path, mode='binary'):
    '''
    Build a dataset of filenames and labels according to the type of classification
    :param cfg:
    :param covid_data_path:
    :param kaggle_data_path:
    :return:
    '''

    # Assemble filenames comprising COVID dataset
    metadata_file_path = covid_data_path + 'metadata.csv'
    covid_df = pd.read_csv(metadata_file_path)
    PA_cxrs_df = (covid_df['view'] == 'PA')
    covid_patients_df = covid_df['finding'] == 'COVID-19'

    if mode == 'binary':
        PA_covid_df = covid_df[covid_patients_df & PA_cxrs_df]      # PA images diagnosed COVID
        PA_covid_df['label'] = 1
        PA_other_df = covid_df[~covid_patients_df & PA_cxrs_df]     # PA images with other diagnoses
        PA_other_df['label'] = 0
        file_df = pd.concat([PA_covid_df[['filename', 'label']], PA_other_df[['filename', 'label']]], axis=0)
        file_df['filename'] = covid_data_path + 'images\\' + file_df['filename'].astype(str)    # Set as absolute paths

        # Assemble filenames comprising Kaggle dataset that is organized into "normal" and "pneumonia" XRs
        normal_xr_filenames = [(kaggle_data_path + 'normal\\' + f) for f in os.listdir(kaggle_data_path + 'normal\\') if
                      os.path.isfile(os.path.join(kaggle_data_path + 'normal\\', f))]
        normal_xr_filenames = normal_xr_filenames[0: ceil(cfg['DATA']['KAGGLE_DATA_FRAC'] * len(normal_xr_filenames))]
        pneum_xr_filenames = [(kaggle_data_path + 'pneumonia\\' + f) for f in os.listdir(kaggle_data_path + 'pneumonia\\') if
                           os.path.isfile(os.path.join(kaggle_data_path + 'pneumonia\\', f))]
        pneum_xr_filenames = pneum_xr_filenames[0: ceil(cfg['DATA']['KAGGLE_DATA_FRAC'] * len(pneum_xr_filenames))]
        other_file_df = pd.DataFrame({'filename': normal_xr_filenames + pneum_xr_filenames, 'label': 0})

        file_df = pd.concat([file_df, other_file_df], axis=0)         # Combine both datasets
    else:
        n_classes = len(cfg['DATA']['CLASSES'])
        class_dict = {cfg['DATA']['CLASSES'][i]: i for i in range(n_classes)} # Map class name to number
        label_dict = {i: cfg['DATA']['CLASSES'][i] for i in range(n_classes)} # Map class name to number
        PA_covid_df = covid_df[covid_patients_df & PA_cxrs_df]  # PA images diagnosed COVID
        PA_covid_df['label'] = class_dict['COVID-19']
        PA_sars_df = covid_df[(covid_df['finding'] == 'SARS') & PA_cxrs_df]   # PA images diagnosed with SARS
        PA_sars_df['label'] = class_dict['viral_pneumonia']                 # Classify SARS with other viral pneumonias
        PA_strep_df = covid_df[(covid_df['finding'] == 'Streptococcus') & PA_cxrs_df]   # PA images diagnosed with Strep
        PA_strep_df['label'] = class_dict['bacterial_pneumonia']            # Classify Strep as bacterial pneumonias
        file_df = pd.concat([PA_covid_df[['filename', 'label']], PA_sars_df[['filename', 'label']],
                             PA_strep_df[['filename', 'label']]], axis=0)
        file_df['filename'] = covid_data_path + 'images\\' + file_df['filename'].astype(str)  # Set as absolute paths

        # Organize some files from Kaggle dataset into "normal", "bacterial pneumonia", "viral pneumonia" XRs
        normal_xr_files = [(kaggle_data_path + 'normal\\' + f) for f in os.listdir(kaggle_data_path + 'normal\\') if
                               os.path.isfile(os.path.join(kaggle_data_path + 'normal\\', f))]
        normal_xr_files = normal_xr_files[0: ceil(cfg['DATA']['KAGGLE_DATA_FRAC'] * len(normal_xr_files))]
        normal_xr_file_df = pd.DataFrame({'filename': normal_xr_files, 'label': class_dict['normal']})
        viral_pneum_xr_files = [(kaggle_data_path + 'pneumonia\\' + f) for f in
                                    os.listdir(kaggle_data_path + 'pneumonia\\') if
                                    os.path.isfile(os.path.join(kaggle_data_path + 'pneumonia\\', f)) and ('virus' in f)]
        viral_pneum_xr_files = viral_pneum_xr_files[0: ceil(cfg['DATA']['KAGGLE_DATA_FRAC'] * len(viral_pneum_xr_files))]
        viral_xr_file_df = pd.DataFrame({'filename': viral_pneum_xr_files, 'label': class_dict['viral_pneumonia']})
        bacterial_pneum_xr_files = [(kaggle_data_path + 'pneumonia\\' + f) for f in
                                    os.listdir(kaggle_data_path + 'pneumonia\\') if
                                    os.path.isfile(os.path.join(kaggle_data_path + 'pneumonia\\', f)) and ('bacteria' in f)]
        bacterial_pneum_xr_files = bacterial_pneum_xr_files[0: ceil(cfg['DATA']['KAGGLE_DATA_FRAC'] * len(bacterial_pneum_xr_files))]
        bacterial_xr_file_df = pd.DataFrame({'filename': bacterial_pneum_xr_files, 'label': class_dict['bacterial_pneumonia']})
        other_file_df = pd.concat([normal_xr_file_df, viral_xr_file_df, bacterial_xr_file_df], axis=0)

        file_df = pd.concat([file_df, other_file_df], axis=0)  # Combine both datasets
        file_df['label_str'] = file_df['label'].map(label_dict) # Add column for string representation of label

    return file_df


def remove_text(img):
    '''
    Attempts to remove textual artifacts from X-ray images. For example, many images indicate the right side of the
    body with a white 'R'. Works only for very bright text.
    :param img: Numpy array of image
    :return: Array of image with (ideally) any characters removed and inpainted
    '''
    mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1][:, :, 0].astype(np.uint8)
    img = img.astype(np.uint8)
    result = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS).astype(np.float32)
    return result


def preprocess(mode='binary'):
    '''
    Preprocess and partition image data. Assemble all image file paths and partition into training, validation and
    test sets. Copy raw images to folders for training, validation and test sets.
    :param mode: Type of classification. Set to either 'binary' or 'multiclass'
    '''

    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))  # Load config data
    covid_data_path = cfg['PATHS']['RAW_COVID_DATA']
    other_data_path = cfg['PATHS']['RAW_OTHER_DATA']
    processed_path = cfg['PATHS']['PROCESSED_DATA']

    # Build dataset based on type of classification
    file_df = build_dataset(cfg, covid_data_path, other_data_path, mode=mode)

    # Split dataset into train, val and test sets
    val_split = cfg['DATA']['VAL_SPLIT']
    test_split = cfg['DATA']['TEST_SPLIT']
    file_df_train, file_df_test = train_test_split(file_df, test_size=test_split, stratify=file_df['label'])
    relative_val_split = val_split / (1 - test_split)  # Calculate fraction of train_df to be used for validation
    file_df_train, file_df_val = train_test_split(file_df_train, test_size=relative_val_split,
                                                      stratify=file_df_train['label'])

    # Delete old datasets
    dest_dir = os.path.join(os.getcwd(), processed_path)
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
    preprocess(mode='multiclass')