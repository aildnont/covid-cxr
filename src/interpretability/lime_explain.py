from lime.lime_image import *
import pandas as pd
import yaml
import os
import datetime
import dill
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.visualization.visualize import visualize_explanation
from src.data.preprocess import remove_text

def predict_instance(x, model):
    '''
    Helper function for LIME explainer. Runs model prediction on perturbations of the example.
    :param x: List of perturbed examples from an example
    :param model: Keras model
    :return: A numpy array comprising a list of class probabilities for each predicted perturbation
    '''
    y = model.predict(x)  # Run prediction on the perturbations
    if y.shape[1] == 1:
        probs = np.concatenate([1.0 - y, y], axis=1)  # Compute class probabilities from the output of the model
    else:
        probs = y
    return probs


def predict_and_explain(x, model, exp, num_features, num_samples, segmentation_fn):
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

    # Generate explanation for the example
    explanation = exp.explain_instance(x, predict, num_features=num_features, num_samples=num_samples, segmentation_fn=segmentation_fn)
    probs = predict_instance(np.expand_dims(x, axis=0), model)
    return explanation, probs


def setup_lime():
    '''
    Load relevant information and create a LIME Explainer
    :return: dict containing important information and objects for explanation experiments
    '''

    # Load relevant constants from project config file
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    lime_dict = {}
    lime_dict['NUM_SAMPLES'] = cfg['LIME']['NUM_SAMPLES']
    lime_dict['NUM_FEATURES'] = cfg['LIME']['NUM_FEATURES']
    lime_dict['IMG_PATH'] = cfg['PATHS']['IMAGES']
    lime_dict['TEST_IMGS_PATH'] = cfg['PATHS']['TEST_IMGS']
    lime_dict['PRED_THRESHOLD'] = cfg['PREDICTION']['THRESHOLD']
    lime_dict['CLASS_MODE'] = cfg['TRAIN']['CLASS_MODE']
    lime_dict['CLASSES'] = cfg['DATA']['CLASSES']
    lime_dict['COVID_ONLY'] = cfg['LIME']['COVID_ONLY']
    KERNEL_WIDTH = cfg['LIME']['KERNEL_WIDTH']
    FEATURE_SELECTION = cfg['LIME']['FEATURE_SELECTION']

    # Load train and test sets
    lime_dict['TRAIN_SET'] = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
    lime_dict['TEST_SET'] = pd.read_csv(cfg['PATHS']['TEST_SET'])

    # Create ImageDataGenerator for test set
    if cfg['TRAIN']['CLASS_MODE'] == 'binary':
        y_col = 'label'
        class_mode = 'raw'
    else:
        y_col = 'label_str'
        class_mode = 'categorical'
    test_img_gen = ImageDataGenerator(rescale=1.0/255.0, preprocessing_function=remove_text)
    test_generator = test_img_gen.flow_from_dataframe(dataframe=lime_dict['TEST_SET'], directory=cfg['PATHS']['TEST_IMGS'],
        x_col="filename", y_col=y_col, target_size=tuple(cfg['DATA']['IMG_DIM']), batch_size=1,
        class_mode=class_mode, shuffle=False)
    lime_dict['TEST_GENERATOR'] = test_generator

    # Define the LIME explainer
    lime_dict['EXPLAINER'] = LimeImageExplainer(kernel_width=KERNEL_WIDTH, feature_selection=FEATURE_SELECTION,
                                                verbose=True)
    dill.dump(lime_dict['EXPLAINER'], open(cfg['PATHS']['LIME_EXPLAINER'], 'wb'))    # Serialize the explainer

    # Load trained model's weights
    lime_dict['MODEL'] = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)

    test_predictions = lime_dict['MODEL'].predict_generator(test_generator, verbose=0)
    return lime_dict


def explain_xray(lime_dict, idx, save_exp=True):
    '''
    # Make a prediction and explain the rationale
    :param lime_dict: dict containing important information and objects for explanation experiments
    :param idx: index of image in test set to explain
    :param save_exp: Boolean indicating whether to save the explanation visualization
    '''

    # Get i'th image in test set
    lime_dict['TEST_GENERATOR'].reset()
    for i in range(idx + 1):
        x, y = lime_dict['TEST_GENERATOR'].next()
    x = np.squeeze(x, axis=0)

    # Algorithm for superpixel segmentation. Parameters set to limit size of superpixels and promote border smoothness
    segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=2.25, max_dist=50, ratio=0.1, sigma=0.15)

    # Make a prediction for this image and retrieve a LIME explanation for the prediction
    start_time = datetime.datetime.now()
    explanation, probs = predict_and_explain(x, lime_dict['MODEL'], lime_dict['EXPLAINER'],
                                      lime_dict['NUM_FEATURES'], lime_dict['NUM_SAMPLES'], segmentation_fn=segmentation_fn)
    print("Explanation time = " + str((datetime.datetime.now() - start_time).total_seconds()) + " seconds")

    # Visualize the LIME explanation and optionally save it to disk
    img_filename = lime_dict['TEST_SET']['filename'][i]
    label = lime_dict['TEST_SET']['label'][i]
    if save_exp:
        file_path = lime_dict['IMG_PATH']
    else:
        file_path = None
    if lime_dict['CLASS_MODE'] == 'multiclass' and lime_dict['COVID_ONLY'] == True:
        label_to_see = 0    # See COVID-19 class explanation
    else:
        label_to_see = 'top'
    visualize_explanation(x, explanation, img_filename, label, probs[0], lime_dict['CLASSES'], label_to_see=label_to_see,
                          file_path=file_path)
    return


if __name__ == '__main__':
    lime_dict = setup_lime()
    i = 0                                                       # Select i'th image in test set
    explain_xray(lime_dict, i, save_exp=True)                  # Generate explanation for image