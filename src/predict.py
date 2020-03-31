import pandas as pd
import yaml
import os
import dill
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from lime.wrappers.scikit_image import SegmentationAlgorithm
from src.data.preprocess import remove_text
from src.visualization.visualize import visualize_explanation


def predict_instance(x, model):
    '''
    Runs model prediction on 1 or more input images.
    :param x: Image(s) to predict
    :param model: A Keras model
    :return: A numpy array comprising a list of class probabilities for each prediction
    '''
    y = model.predict(x)  # Run prediction on the perturbations
    if y.shape[1] == 1:
        probs = np.concatenate([1.0 - y, y], axis=1)  # Compute class probabilities from the output of the model
    else:
        probs = y
    return probs


def predict_and_explain(x, model, exp, num_features, num_samples):
    '''
    Use the model to predict a single example and apply LIME to generate an explanation.
    :param x: Preprocessed image to predict
    :param model: The trained neural network model
    :param exp: A LimeImageExplainer object
    :param num_features: # of features to use in explanation
    :param num_samples: # of times to perturb the example to be explained
    :return: The LIME explainer for the instance
    '''

    def predict(x):
        '''
        Helper function for LIME explainer. Runs model prediction on perturbations of the example.
        :param x: List of perturbed examples from an example
        :return: A numpy array constituting a list of class probabilities for each predicted perturbation
        '''
        probs = predict_instance(x, model)
        return probs

    # Algorithm for superpixel segmentation. Parameters set to limit size of superpixels and promote border smoothness
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=2.25, max_dist=50, ratio=0.1, sigma=0.15)

    # Generate explanation for the example
    explanation = exp.explain_instance(x, predict, num_features=num_features, num_samples=num_samples, segmentation_fn=segmentation_fn)
    probs = predict_instance(np.expand_dims(x, axis=0), model)
    return explanation, probs


def predict_and_explain_set(raw_img_dir=None, preds_dir=None, save_results=True, give_explanations=True):
    '''
    Preprocess a raw dataset. Then get model predictions and corresponding explanations.
    :param raw_img_dir: Directory in which to look for raw images
    :param preds_dir: Path at which to save results of this prediction
    :param save_results: Flag specifying whether to save the prediction results to disk
    :param give_explanations: Flag specifying whether to provide LIME explanations with predictions spreadsheet
    :return: Dataframe of prediction results, optionally including explanations.
    '''

    # Load project config data
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    cur_date = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Restore the model, LIME explainer, and model class indices from their respective serializations
    model = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)
    explainer = dill.load(open(cfg['PATHS']['LIME_EXPLAINER'], 'rb'))
    class_indices = dill.load(open(cfg['PATHS']['OUTPUT_CLASS_INDICES'], 'rb'))

    # Load LIME and prediction constants from config
    NUM_SAMPLES = cfg['LIME']['NUM_SAMPLES']
    NUM_FEATURES = cfg['LIME']['NUM_FEATURES']
    CLASS_NAMES = cfg['DATA']['CLASSES']

    # Define column names of the DataFrame representing the prediction results
    col_names = ['Image Filename', 'Predicted Class']
    for c in cfg['DATA']['CLASSES']:
        col_names.append('p(' + c + ')')

    # Add columns for client explanation
    if give_explanations:
        col_names.append('Explanation Filename')

    # Set raw image directory based on project config, if not specified
    if raw_img_dir is None:
        raw_img_dir = cfg['PATHS']['BATCH_PRED_IMGS']

    # If no path is specified, create new directory for predictions
    if preds_dir is None:
        preds_dir = cfg['PATHS']['BATCH_PREDS'] + '\\' + cur_date + '\\'
        if save_results and not os.path.exists(cfg['PATHS']['BATCH_PREDS'] + '\\' + cur_date):
            os.mkdir(preds_dir)

    # Create DataFrame for raw image file names
    raw_img_df = pd.DataFrame({'filename': os.listdir(raw_img_dir)})
    raw_img_df = raw_img_df[raw_img_df['filename'].str.contains('jpg|png|jpeg', na=False)]   # Enforce image files

    # Create generator for the image files
    img_gen = ImageDataGenerator(preprocessing_function=remove_text, samplewise_std_normalization=True,
                                 samplewise_center=True)
    img_iter = img_gen.flow_from_dataframe(dataframe=raw_img_df, directory=raw_img_dir, x_col="filename",
                                           target_size=cfg['DATA']['IMG_DIM'], batch_size=1, class_mode=None,
                                           shuffle=False)

    # Predict (and optionally explain) all images in the specified directory
    rows = []
    print('Predicting and explaining examples.')

    for filename in raw_img_df['filename'].tolist():

        # Get preprocessed image and make a prediction.
        try:
            x = img_iter.next()
        except StopIteration:
            break
        y = np.squeeze(predict_instance(x, model))

        # Rearrange prediction probability vector to reflect original ordering of classes in project config
        p = [y[CLASS_NAMES.index(c)] for c in class_indices]
        predicted_class = CLASS_NAMES[np.argmax(p)]
        row = [filename, predicted_class]
        row.extend(list(p))

        # Explain this prediction
        if give_explanations:
            explanation, _ = predict_and_explain(np.squeeze(x, axis=0), model, explainer, NUM_FEATURES, NUM_SAMPLES)
            if cfg['LIME']['COVID_ONLY'] == True:
                label_to_see = class_indices['COVID-19']
            else:
                label_to_see = 'top'

            # Load and resize the corresponding original image (no preprocessing)
            orig_img = cv2.imread(raw_img_dir + filename)
            orig_img = cv2.resize(orig_img, tuple(cfg['DATA']['IMG_DIM']), interpolation=cv2.INTER_NEAREST)

            # Generate visual for explanation
            exp_filename = visualize_explanation(orig_img, explanation, filename, None, p, CLASS_NAMES,
                                                 label_to_see=label_to_see, file_path=preds_dir)
            row.append(exp_filename.split('\\')[-1])
        rows.append(row)

    # Convert results to a Pandas DataFrame and save
    results_df = pd.DataFrame(rows, columns=col_names)
    if save_results:
        results_path = preds_dir + 'predictions.csv'
        results_df.to_csv(results_path, columns=col_names, index_label=False, index=False)
    return results_df


if __name__ == '__main__':
    results = predict_and_explain_set(preds_dir=None, save_results=True, give_explanations=True)