import numpy as np
import pandas as pd
import pydicom as dicom
import yaml
import os
import pathlib
import shutil
import cv2
from math import ceil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def build_dataset(cfg):
    '''
    Build a dataset of filenames and labels according to the type of classification
    :param cfg: Project config dictionary
    :return: DataFrame of file names of examples and corresponding class labels
    '''

    # Get paths of raw datasets to be included
    covid_data_path = cfg['PATHS']['RAW_COVID_DATA']
    other_data_path = cfg['PATHS']['RAW_OTHER_DATA']

    # Assemble filenames comprising COVID dataset
    metadata_file_path = covid_data_path + 'metadata.csv'
    covid_df = pd.read_csv(metadata_file_path)
    covid_PA_cxrs_df = (covid_df['view'] == 'PA')
    covid_patients_df = (covid_df['finding'] == 'COVID-19')

    # Assemble filenames comprising RSNA dataset
    rsna_metadata_path = other_data_path + 'stage_2_train_labels.csv'
    rsna_df = pd.read_csv(rsna_metadata_path)
    num_rsna_imgs = cfg['DATA']['NUM_RSNA_IMGS']
    rsna_normal_df = rsna_df[rsna_df['Target'] == 0][0:num_rsna_imgs//2]
    rsna_pneum_df = rsna_df[rsna_df['Target'] == 1][0:num_rsna_imgs//2]

    # Convert dicom files to jpg if not done already in a previous run
    for filename in tqdm(rsna_normal_df['patientId'].tolist()):
        if not os.path.exists(other_data_path + filename + '.jpg'):
            ds = dicom.dcmread(os.path.join(other_data_path + 'stage_2_train_images/' + filename + '.dcm'))
            pixel_arr = ds.pixel_array
            cv2.imwrite(os.path.join(other_data_path + filename + '.jpg'), pixel_arr)
    for filename in tqdm(rsna_pneum_df['patientId'].tolist()):
        if not os.path.exists(other_data_path + filename + '.jpg'):
            ds = dicom.dcmread(os.path.join(other_data_path + 'stage_2_train_images/' + filename + '.dcm'))
            pixel_arr = ds.pixel_array
            cv2.imwrite(os.path.join(other_data_path + filename + '.jpg'), pixel_arr)

    mode = cfg['TRAIN']['CLASS_MODE']
    n_classes = len(cfg['DATA']['CLASSES'])
    class_dict = {cfg['DATA']['CLASSES'][i]: i for i in range(n_classes)}  # Map class name to number
    label_dict = {i: cfg['DATA']['CLASSES'][i] for i in range(n_classes)}  # Map class name to number

    if mode == 'binary':
        PA_covid_df = covid_df[covid_patients_df & covid_PA_cxrs_df]      # PA images diagnosed COVID
        PA_covid_df['label'] = 1
        PA_other_df = covid_df[~covid_patients_df & covid_PA_cxrs_df]     # PA images with other diagnoses
        PA_other_df['label'] = 0
        file_df = pd.concat([PA_covid_df[['filename', 'label']], PA_other_df[['filename', 'label']]], axis=0)
        file_df['filename'] = covid_data_path + 'images/' + file_df['filename'].astype(str)    # Set as absolute paths

        rsna_df = pd.concat([rsna_normal_df, rsna_pneum_df], axis=0)
        rsna_filenames = other_data_path + rsna_df['patientId'].astype(str) + '.jpg'
        rsna_file_df = pd.DataFrame({'filename': rsna_filenames, 'label': 0})

        file_df = pd.concat([file_df, rsna_file_df], axis=0)         # Combine both datasets
    else:
        PA_covid_df = covid_df[covid_patients_df & covid_PA_cxrs_df]  # PA images diagnosed COVID
        PA_covid_df['label'] = class_dict['COVID-19']
        PA_pneum_df = covid_df[covid_df['finding'].isin(['SARS', 'Steptococcus']) & covid_PA_cxrs_df]   # PA images diagnosed with SARS
        PA_pneum_df['label'] = class_dict['other_pneumonia']                 # Classify SARS with other viral pneumonias
        file_df = pd.concat([PA_covid_df[['filename', 'label']], PA_pneum_df[['filename', 'label']]], axis=0)
        file_df['filename'] = covid_data_path + 'images/' + file_df['filename'].astype(str)  # Set as absolute paths

        # Organize some files from RSNA dataset into "normal", and "pneumonia" XRs
        rsna_normal_filenames = other_data_path + rsna_normal_df['patientId'].astype(str) + '.jpg'
        rsna_pneum_filenames = other_data_path + rsna_pneum_df['patientId'].astype(str) + '.jpg'
        rsna_normal_file_df = pd.DataFrame({'filename': rsna_normal_filenames, 'label': class_dict['normal']})
        rsna_pneum_file_df = pd.DataFrame({'filename': rsna_pneum_filenames, 'label': class_dict['other_pneumonia']})
        rsna_file_df = pd.concat([rsna_normal_file_df, rsna_pneum_file_df], axis=0)

        file_df = pd.concat([file_df, rsna_file_df], axis=0)  # Combine both datasets

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


def mean_scale(img):
    '''
    Scale the image according to its mean and standard deviation
    :param img: Numpy array of image
    :return:
    '''
    return ((img - np.mean(img)) / np.std(img)).astype(np.float32)


def transform_img(img):
    '''
    Apply custom transformation to a single image
    :param img: Numpy array of image
    :return:
    '''
    img = remove_text(img)
    img = mean_scale(img)
    return img


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


def preprocess():
    '''
    Preprocess and partition image data. Assemble all image file paths and partition into training, validation and
    test sets. Copy raw images to folders for training, validation and test sets.
    :param mode: Type of classification. Set to either 'binary' or 'multiclass'
    '''

    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))  # Load config data
    processed_path = cfg['PATHS']['PROCESSED_DATA']

    # Build dataset based on type of classification
    file_df = build_dataset(cfg)

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
    for path in pathlib.Path(os.path.join(dest_dir, 'train/')).glob('*'):
        if '.gitkeep' not in str(path):
            path.unlink()
    for path in pathlib.Path(os.path.join(dest_dir, 'val/')).glob('*'):
        if '.gitkeep' not in str(path):
            path.unlink()
    for path in pathlib.Path(os.path.join(dest_dir, 'test/')).glob('*'):
        if '.gitkeep' not in str(path):
            path.unlink()

    # Copy images to appropriate directories
    print('Copying training set images.')
    for file_path in tqdm(file_df_train['filename'].tolist()):
        shutil.copy(file_path, os.path.join(dest_dir, 'train/'))
    print('Copying validation set images.')
    for file_path in tqdm(file_df_val['filename'].tolist()):
        shutil.copy(file_path, os.path.join(dest_dir, 'val/'))
    print('Copying test set images.')
    for file_path in tqdm(file_df_test['filename'].tolist()):
        shutil.copy(file_path, os.path.join(dest_dir, 'test/'))

    # Update file path dataframes
    file_df_train['filename'] = file_df_train['filename'].str.split('/').str[-1]
    file_df_val['filename'] = file_df_val['filename'].str.split('/').str[-1]
    file_df_test['filename'] = file_df_test['filename'].str.split('/').str[-1]

    # Save training, validation and test sets
    file_df_train.to_csv(cfg['PATHS']['TRAIN_SET'])
    file_df_val.to_csv(cfg['PATHS']['VAL_SET'])
    file_df_test.to_csv(cfg['PATHS']['TEST_SET'])
    return

if __name__ == '__main__':
    preprocess()