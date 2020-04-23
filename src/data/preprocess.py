import numpy as np
import pandas as pd
import pydicom as dicom
import yaml
import os
import pathlib
import shutil
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def build_dataset(cfg):
    '''
    Build a dataset of filenames and labels according to the type of classification
    :param cfg: Project config dictionary
    :return: DataFrame of file names of examples and corresponding class labels
    '''

    # Get paths of raw datasets to be included
    mila_data_path = cfg['PATHS']['MILA_DATA']
    fig1_data_path = cfg['PATHS']['FIGURE1_DATA']
    rsna_data_path = cfg['PATHS']['RSNA_DATA']

    # Assemble filenames comprising Mila dataset
    mila_df = pd.read_csv(mila_data_path + 'metadata.csv')
    mila_df['filename'] = mila_data_path.split('/')[-2] + '/images/' + mila_df['filename'].astype(str)
    mila_views_cxrs_df = (mila_df['view'].str.contains('|'.join(cfg['DATA']['VIEWS'])))    # Select desired X-ray views
    mila_covid_pts_df = (mila_df['finding'] == 'COVID-19')
    mila_covid_views_df = mila_df[mila_covid_pts_df & mila_views_cxrs_df]  # Images for patients diagnosed with COVID-19

    # Assemble filenames comprising Figure 1 dataset
    fig1_df = pd.read_csv(fig1_data_path + 'metadata.csv', encoding='ISO-8859-1')
    fig1_df['filename'] = ''
    for i, row in fig1_df.iterrows():
        if os.path.exists(fig1_data_path + 'images/' + fig1_df.loc[i, 'patientid'] + '.jpg'):
            fig1_df.loc[i, 'filename'] = fig1_data_path.split('/')[-2] + '/images/' + fig1_df.loc[i, 'patientid'] + '.jpg'
        else:
            fig1_df.loc[i, 'filename'] = fig1_data_path.split('/')[-2] + '/images/' + fig1_df.loc[i, 'patientid'] + '.png'
    fig1_df['view'].fillna('PA or AP', inplace=True)    # All images in this dataset are either AP or PA
    fig1_views_cxrs_df = (fig1_df['view'].str.contains('|'.join(cfg['DATA']['VIEWS'])))    # Select desired X-ray views
    fig1_covid_pts_df = (fig1_df['finding'] == 'COVID-19')
    fig1_covid_views_df = fig1_df[fig1_covid_pts_df & fig1_views_cxrs_df]  # Images for patients diagnosed COVID-19

    # Assemble filenames comprising RSNA dataset
    rsna_metadata_path = rsna_data_path + 'stage_2_train_labels.csv'
    rsna_df = pd.read_csv(rsna_metadata_path)
    num_rsna_imgs = cfg['DATA']['NUM_RSNA_IMGS']
    rsna_normal_df = rsna_df[rsna_df['Target'] == 0]
    rsna_pneum_df = rsna_df[rsna_df['Target'] == 1]

    # Convert dicom files of CXRs with no findings to jpg if not done already in a previous run. Select desired views.
    file_counter = 0
    normal_idxs = []
    for df_idx in rsna_normal_df.index.values.tolist():
        filename = rsna_normal_df.loc[df_idx]['patientId']
        ds = dicom.dcmread(os.path.join(rsna_data_path + 'stage_2_train_images/' + filename + '.dcm'))
        if any(view in ds.SeriesDescription.split(' ')[1] for view in cfg['DATA']['VIEWS']):  # Select desired X-ray views
            if not os.path.exists(rsna_data_path + filename + '.jpg'):
                cv2.imwrite(os.path.join(rsna_data_path + filename + '.jpg'), ds.pixel_array)   # Save as .jpg
            normal_idxs.append(df_idx)
            file_counter += 1
        if file_counter >= num_rsna_imgs // 2:
            break
    rsna_normal_df = rsna_normal_df.loc[normal_idxs]

    # Convert dicom files of CXRs with pneumonia to jpg if not done already in a previous run. Select desired views.
    file_counter = 0
    pneum_idxs = []
    num_remaining = num_rsna_imgs - num_rsna_imgs // 2
    for df_idx in rsna_pneum_df.index.values.tolist():
        filename = rsna_pneum_df.loc[df_idx]['patientId']
        ds = dicom.dcmread(os.path.join(rsna_data_path + 'stage_2_train_images/' + filename + '.dcm'))
        if any(view in ds.SeriesDescription.split(' ')[1] for view in cfg['DATA']['VIEWS']):  # Select desired X-ray views
            if not os.path.exists(rsna_data_path + filename + '.jpg'):
                cv2.imwrite(os.path.join(rsna_data_path + filename + '.jpg'), ds.pixel_array)  # Save as .jpg
            pneum_idxs.append(df_idx)
            file_counter += 1
        if file_counter >= num_remaining:
            break
    rsna_pneum_df = rsna_pneum_df.loc[pneum_idxs]

    mode = cfg['TRAIN']['CLASS_MODE']
    n_classes = len(cfg['DATA']['CLASSES'])
    class_dict = {cfg['DATA']['CLASSES'][i]: i for i in range(n_classes)}  # Map class name to number
    label_dict = {i: cfg['DATA']['CLASSES'][i] for i in range(n_classes)}  # Map class name to number

    if mode == 'binary':
        mila_covid_views_df['label'] = 1                                       # Mila images with COVID-19 diagnosis
        mila_other_views_df = mila_df[~mila_covid_pts_df & mila_views_cxrs_df]
        mila_other_views_df['label'] = 0                                       # Mila images with alternative diagnoses
        fig1_covid_views_df['label'] = 1                                       # Figure 1 images with COVID-19 diagnosis
        file_df = pd.concat([mila_covid_views_df[['filename', 'label']], mila_other_views_df[['filename', 'label']],
                             fig1_covid_views_df[['filename', 'label']]], axis=0)

        rsna_df = pd.concat([rsna_normal_df, rsna_pneum_df], axis=0)
        rsna_filenames = rsna_data_path.split('/')[-2] + '/' + rsna_df['patientId'].astype(str) + '.jpg'
        rsna_file_df = pd.DataFrame({'filename': rsna_filenames, 'label': 0})

        file_df = pd.concat([file_df, rsna_file_df], axis=0)         # Combine both datasets
    else:
        mila_covid_views_df['label'] = class_dict['COVID-19']
        mila_views_pneum_df = mila_df[mila_df['finding'].isin(['SARS', 'Steptococcus', 'MERS', 'Legionella', 'Klebsiella',
                                                            'Chlamydophila', 'Pneumocystis']) & mila_views_cxrs_df]
        mila_views_pneum_df['label'] = class_dict['other_pneumonia']                 # Mila CXRs with other peumonias
        mila_views_normal_df = mila_df[mila_df['finding'].isin(['No finding']) & mila_views_cxrs_df]
        mila_views_normal_df['label'] = class_dict['normal']                         # Mila CXRs with no finding
        fig1_covid_views_df['label'] = class_dict['COVID-19']                        # Figure 1 CXRs with COVID-19 finding
        file_df = pd.concat([mila_covid_views_df[['filename', 'label']], mila_views_pneum_df[['filename', 'label']],
                             mila_views_normal_df[['filename', 'label']], fig1_covid_views_df[['filename', 'label']]], axis=0)

        # Organize some files from RSNA dataset into "normal", and "pneumonia" XRs
        rsna_normal_filenames = rsna_data_path.split('/')[-2] + '/' + rsna_normal_df['patientId'].astype(str) + '.jpg'
        rsna_pneum_filenames = rsna_data_path.split('/')[-2] + '/' + rsna_pneum_df['patientId'].astype(str) + '.jpg'
        rsna_normal_file_df = pd.DataFrame({'filename': rsna_normal_filenames, 'label': class_dict['normal']})
        rsna_pneum_file_df = pd.DataFrame({'filename': rsna_pneum_filenames, 'label': class_dict['other_pneumonia']})
        rsna_file_df = pd.concat([rsna_normal_file_df, rsna_pneum_file_df], axis=0)

        file_df = pd.concat([file_df, rsna_file_df], axis=0)  # Combine both datasets

    file_df['label_str'] = file_df['label'].map(label_dict) # Add column for string representation of label
    return file_df


def remove_text(img):
    '''
    Attempts to remove bright textual artifacts from X-ray images. For example, many images indicate the right side of
    the body with a white 'R'. Works only for very bright text.
    :param img: Numpy array of image
    :return: Array of image with (ideally) any characters removed and inpainted
    '''
    mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)[1][:, :, 0].astype(np.uint8)
    img = img.astype(np.uint8)
    result = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS).astype(np.float32)
    return result


def preprocess(cfg=None):
    '''
    Preprocess and partition image data. Assemble all image file paths and partition into training, validation and
    test sets. Copy raw images to folders for training, validation and test sets.
    :param cfg: Optional parameter to set your own config object.
    '''

    if cfg is None:
        cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))  # Load config data

    # Build dataset based on type of classification
    file_df = build_dataset(cfg)

    # Split dataset into train, val and test sets
    val_split = cfg['DATA']['VAL_SPLIT']
    test_split = cfg['DATA']['TEST_SPLIT']
    file_df_train, file_df_test = train_test_split(file_df, test_size=test_split, stratify=file_df['label'])
    relative_val_split = val_split / (1 - test_split)  # Calculate fraction of train_df to be used for validation
    file_df_train, file_df_val = train_test_split(file_df_train, test_size=relative_val_split,
                                                      stratify=file_df_train['label'])

    # Save training, validation and test sets
    if not os.path.exists(cfg['PATHS']['PROCESSED_DATA']):
        os.makedirs(cfg['PATHS']['PROCESSED_DATA'])
    file_df_train.to_csv(cfg['PATHS']['TRAIN_SET'])
    file_df_val.to_csv(cfg['PATHS']['VAL_SET'])
    file_df_test.to_csv(cfg['PATHS']['TEST_SET'])
    return

if __name__ == '__main__':
    preprocess()