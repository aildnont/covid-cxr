# COVID-19 Chest X-Ray Model
![alt text](documents/readme_images/london_logo.png "City of London logo")

The goals of this project are threefold: (1) to initially develop a machine learning
algorithm to distinguish chest X-rays of individuals with respiratory
illness testing positive for COVID-19 from other X-rays, (2) to
promote discovery of patterns in such X-rays via machine learning
interpretability algorithms, and (3) to build more robust and extensible machine learning infrastructure trained on a variety of data types, to aid in the clobal response to COVID-19. 

We are calling on all machine learning
practitioners and healthcare professionals who can contribute their expertise
to this effort. If you are interested in getting involved in this project by lending your expertise, [sign up here](https://forms.gle/6Qo34h4DsUrRJNVR9), otherwise, feel free to experiment with the code base in this repo.

A model has been trained on a dataset composed of X-rays
labeled positive for COVID-19 infection, normal X-rays, and X-rays
depicting evidence of other pneumonias. Currently, we are using
[Local Interpretable
Model-Agnostic Explanations](https://arxiv.org/pdf/1602.04938.pdf) (i.e.
LIME) as the interpretability method being applied to the model. This
project is in need of more expertise and more data. Please consider
contributing or reaching out to us if you are able to lend a hand by signing up at the above link. This
project is in its infancy. The immediacy of this work cannot be
overstated, as any insights derived from this project may be of benefit
to healthcare practitioners and researchers as the COVID-19 pandemic
continues to evolve.

## Why X-rays?
There have been promising efforts to apply machine learning to aid in
the diagnosis of COVID-19 based on
[CT scans](https://pubs.rsna.org/doi/10.1148/radiol.2020200905). Despite
the success of these methods, the fact remains that COVID-19 is an
infection that is likely to be experienced by communities of all sizes.
X-rays are inexpensive and quick to perform; therefore, they are more
accessible to healthcare providers working in smaller and/or remote
regions. Any insights that may be derived as a result of explainability
algorithms applied to a successful model will be invaluable to the
global effort of identifying and treating cases of COVID-19. This model is a prototype system and not for medical use and does not offer a diagnosis. 

## Getting Started
1. Clone this repository (for help see this
   [tutorial](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository)).
2. Install the necessary dependencies (listed in
   [requirements.txt](requirements.txt)). To do this, open a terminal in
   the root directory of the project and run the following:
   ```
   $ pip install -r requirements.txt
   ```
3. Clone the
   [covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset)
   repository somewhere on your local machine. Set the _RAW_COVID_DATA_
   field in the _PATHS_ section of [config.yml](config.yml) to the
   address of the root directory of the cloned repository (for help see
   [Project Config](#project-config)).
4. Download and unzip the
   [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
   dataset from Kaggle somewhere on your local machine. Set the
   _RAW_OTHER_DATA_ field in the _PATHS_ section of
   [config.yml](config.yml) to the address of the folder containing the
   dataset.
5. Execute [_preprocess.py_](src/data/preprocess.py) to create Pandas
   DataFrames of filenames and labels. Preprocessed DataFrames and
   corresponding images of the dataset will be saved within
   _data/preprocessed/_.
6. Execute [_train.py_](src/train.py) to train the neural network model.
   The trained model weights will be saved within _results/models/_, and
   its filename will resemble the following structure:
   modelyyyymmdd-hhmmss.h5, where yyyymmdd-hhmmss is the current time.
   The [TensorBoard](https://www.tensorflow.org/tensorboard) log files
   will be saved within _results/logs/training/_.
7. In [config.yml](config.yml), set _MODEL_TO_LOAD_ within _PATHS_ to
   the path of the model weights file that was generated in step 6 (for
   help see [Project Config](#project-config)). Execute
   [_lime_explain.py_](src/interpretability/lime_explain.py) to generate
   interpretable explanations for the model's predictions on the test
   set. See more details in the [LIME Section](#lime).

## Train a model and visualize results
1. Once you have the appropriate datasets downloaded, execute
   [_preprocess.py_](src/data/preprocess.py). See
   [Getting Started](#getting-started) for help obtaining and organizing
   these raw image datasets. If this script ran properly, you should see
   folders entitled _train_, _test_, and _val_ within
   _data/preprocessed_. Addionally, you should see 3 files entitled
   _train_set.csv_, _val_set.csv_, and _test_set.csv_.
2. In [config.yml](config.yml), ensure that _EXPERIMENT_TYPE_ within
   _TRAIN_ is set to _'single_train'_.
3. Execute [train.py](src/train.py). The trained model's weights will be
   located in _results/models/_, and its filename will resemble the
   following structure: modelyyyymmdd-hhmmss.h5, where yyyymmdd-hhmmss
   is the current time. The model's logs will be located in
   _results/logs/training/_, and its directory name will be the current
   time in the same format. These logs contain information about the
   experiment, such as metrics throughout the training process on the
   training and validation sets, and performance on the test set. The
   logs can be visualized by running
   [TensorBoard](https://www.tensorflow.org/tensorboard) locally. See
   below for an example of a plot from a TensorBoard log file depicting
   loss on the training and validation sets versus epoch. Plots
   depicting the change in performance metrics throughout the training
   process (such as the example below) are available in the _SCALARS_
   tab of TensorBoard.  
   ![alt text](documents/readme_images/tensorboard_loss.png "Loss vs
   Epoch")  
   You can also visualize the trained model's performance on the test
   set. See below for an example of the ROC Curve and Confusion Matrix
   based on test set predictions. In our implementation, these plots are
   available in the _IMAGES_ tab of TensorBoard.  
   ![alt text](documents/readme_images/roc_example.PNG "ROC Curve")
   ![alt text](documents/readme_images/cm_example.PNG "Confusion
   Matrix")

## Binary vs. Multi-class Models
The documentation in this README assumes the user is training a binary
classifier, which is set by default in [config.yml](config.yml). The
user has the option of training a model to perform binary prediction on
whether the X-ray exhibits signs of COVID-19 or training a model to
perform multi-class classification to distinguish COVID-19 cases, other
pneumonia cases and normal X-rays. For a multi-class model, the output
layer is a vector of probabilities outputted by a softmax final layer.
In the multi-class scenario, precision, recall and F1-Score are
calculated for the COVID-19 class only. To train a multi-class
classifier, the user should be aware of the following changes that may
be made to [config.yml](config.yml):
- Within the _TRAIN_ section, set the _CLASS_MODE_ field to
  _'multiclass'_. By default, it is set to _'binary'_.
- The class names are listed in the _CLASSES_ field of the _DATA_
  section.
- The relative weight of classes can be modified by updating the
  _CLASS_MULTIPLIER_ field in the _TRAIN_ section.
- You can update hyperparameters for the multiclass classification model
  by setting the fields in the _DCNN_MULTICLASS_ subsection of _NN_.
- By default, LIME explanations are given for the class of the model's
  prediction. If you wish to generate LIME explanations for the COVID-19
  class only, set _COVID_ONLY_ within _LIME_ to _'true'_.

## LIME Explanations
Since the predictions made by this model may be used be healthcare
providers to benefit patients, it is imperative that the model's
predictions may be explained so as to ensure that the it is making
responsible predictions. Model explainability promotes transparency and
accountability of decision-making algorithms. Since this model is a
neural network, it is difficult to decipher which rules or heuristics it
is employing to make its predictions. Since so little is known about
presentation of COVID-19, interpretability is all the more important. We
used [Local Interpretable
Model-Agnostic Explanations](https://arxiv.org/pdf/1602.04938.pdf) (i.e.
LIME) to explain the predictions of the neural network classifier that
we trained. We used the implementation available in the authors' [GitHub
repository](https://github.com/marcotcr/lime). LIME perturbs the
features in an example and fits a linear model to approximate the neural
network at the local region in the feature space surrounding the
example. It then uses the linear model to determine which features were
most contributory to the model's prediction for that example. By
applying LIME to our trained model, we can conduct informed feature
engineering based on any obviously inconsequential features we see
insights from domain experts. For example, we noticed that different
characters present on normal X-rays were contributing to predictions
based off LIME explanations. To counter this unwanted behaviour, we have
taken steps to remove and inpaint textual regions as much as possible.
See the steps below to apply LIME to explain the model's predictions on
examples in the test set.
1. Having previously run _[preprocess.py](src/data/preprocess.py)_ and
   _[train.py](src/train.py)_, ensure that _data/processed/_ contains
   _Test_Set.csv_ and a folder called _test_ that contains the test set
   images.
2. In [config.yml](config.yml), set _MODEL_TO_LOAD_ within _PATHS_ to
   the path of the model weights file (_.h5_ file) that you wish to use
   for prediction.
3. Execute _[lime_explain.py](src/interpretability/lime_explain.py)_. To
   generate explanations for different images in the test set, modify
   the following call: `explain_xray(lime_dict, i, save_exp=True)`. Set
   _i_ to the index of the test set image you would like to explain and
   rerun the script. If you are using an interactive console, you may
   choose to simply call the function again instead of rerunning the
   script.
4. Interpret the output of the LIME explainer. An image will have been
   generated that depicts the superpixels (i.e. image regions) that were
   most contributory to the model's prediction. Superpixels that
   contributed toward a prediction of COVID-19 are coloured green and
   superpixels that contributed against a prediction of COVID-19 are
   coloured red. The image will be automatically saved in
   _documents/generated_images/_, and its filename will resemble the
   following: _Client_client_id_exp_yyyymmdd-hhmmss.png_. See below for
   examples of this graphic.

It is our hope that healthcare professionals will be able to provide
feedback on the model based on their assessment of the quality of these
explanations. If the explanations make sense to individuals with
extensive experience interpreting X-rays, perhaps certain patterns can
be identified as radiological signatures of COVID-19.

Below are examples of LIME explanations. The top two images are
explanations of a couple of the binary classifier's predictions. Green
regions and red regions identify superpixels that most contributed to
and against prediction of COVID-19 respectively. The bottom two images
are explanations of a couple of the multi-class classifier's
predictions. Green regions and red regions identify superpixels that
most contributed to and against the predicted class respectively.

![alt text](documents/readme_images/LIME_example0.PNG "Sample LIME
explanation #1")  
![alt text](documents/readme_images/LIME_example1.PNG "Sample LIME
explanation #2")  
![alt text](documents/readme_images/LIME_example2.PNG "Sample LIME
explanation #3")  
![alt text](documents/readme_images/LIME_example3.PNG "Sample LIME
explanation #4")

## Project Structure
The project looks similar to the directory structure below. Disregard
any _.gitkeep_ files, as their only purpose is to force Git to track
empty directories. Disregard any _.\__init\__.py_ files, as they are
empty files that enable Python to recognize certain directories as
packages.

```
├── data
│   ├── interpretability          <- Generated feature information
│   └── processed                 <- Products of preprocessing
|
├── documents
|   ├── generated_images          <- Visualizations of model performance, experiments
|   └── readme_images             <- Image assets for README.md
├── results
│   ├── logs                      <- TensorBoard logs
│   └── models                    <- Trained model weights
|
├── src
│   ├── custom                    <- Custom TensorFlow components
|   |   └── metrics.py            <- Definition of custom TensorFlow metrics
│   ├── data                      <- Data processing
|   |   └── preprocess.py         <- Main preprocessing script
│   ├── interpretability          <- Model interpretability scripts
|   |   └── lime_explain.py       <- Script for generating LIME explanations
│   ├── models                    <- TensorFlow model definitions
|   |   └── models.py             <- Script containing model definition
|   ├── visualization             <- Visualization scripts
|   |   └── visualize.py          <- Script for visualizing model performance metrics
|   └── train.py                  <- Script for training model on preprocessed data
|
├── .gitignore                    <- Files to be be ignored by git.
├── config.yml                    <- Values of several constants used throughout project
├── LICENSE                       <- Project license
├── README.md                     <- Project description
└── requirements.txt              <- Lists all dependencies and their respective versions
```

## Project Config
Many of the components of this project are ready for use on your X-ray
datasets. However, this project contains several configurable variables that
are defined in the project config file: [config.yml](config.yml). When
loaded into Python scripts, the contents of this file become a
dictionary through which the developer can easily access its members.

For user convenience, the config file is organized into major steps in
our model development pipeline. Many fields need not be modified by the
typical user, but others may be modified to suit the user's specific
goals. A summary of the major configurable elements in this file is
below.

#### PATHS
- **RAW_COVID_DATA**: Path to folder containing
  [COVID-19 image dataset](https://github.com/ieee8023/covid-chestxray-dataset)
- **RAW_OTHER_DATA**: Path to folder containing
  [RSNA Pneumonia Detection Challenge dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- **MODEL_WEIGHTS**: Path at which to save trained model's weights
- **MODEL_TO_LOAD**: Path to the trained model's weights that you would
  like to load for prediction
#### DATA
- **IMG_DIM**: Desired target size of image after preprocessing
- **VAL_SPLIT**: Fraction of the data allocated to the validation set
- **TEST_SPLIT**: Fraction of the data allocated to the test set
- **KAGGLE_DATA_FRAC**: Fraction of the images from the RSNA Kaggle
  chest X-ray dataset to use. The default value results in a dataset of
  about 1000 images total.
- **CLASSES**: This is an ordered list of class names. Must be the same
  length as the number of classes you wish to distinguish.
#### TRAIN
- **CLASS_MODE**: The type of classification to be performed. Should be
  set before performing preprocessing. Set to either _'binary'_ or
  _'multiclass'_.
- **EXPERIMENT_TYPE**: The type of training experiment you would like to
  perform if executing [_train.py_](src/train.py). For now, the only
  choice is _'single_train'_.
- **BATCH_SIZE**: Mini-batch size during training
- **EPOCHS**: Number of epochs to train the model for
- **THRESHOLDS**: A single float or list of floats in range [0, 1]
  defining the classification threshold. Affects precision and recall
  metrics.
- **PATIENCE**: Number of epochs to wait before freezing the model if
  validation loss does not decrease.
- **IMB_STRATEGY**: Class imbalancing strategy to employ. In our
  dataset, the ratio of positive to negative ground truth was very low,
  prompting the use of these strategies. Set either to _'class_weight'_
  or _'random_oversample'_.
- **CLASS_MULTIPLIER**: A list of coefficients to multiply the computed
  class weights by during computation of loss function. Must be the same
  length as the number of classes.
#### NN
- **DCNN_BINARY**: Contains definitions of configurable hyperparameters
  associated with a custom deep convolutional neural network for binary
  classification. The values currently in this section were the optimal
  values for our dataset informed by heuristically selecting
  hyperparameters.
  - **KERNEL_SIZE**: Kernel size for convolutional layers
  - **STRIDES**: Size of strides for convolutional layers
  - **INIT_FILTERS**: Number of filters for first convolutional layer
  - **FILTER_EXP_BASE**: Base of exponent that determines number of
    filters in successive convolutional layers. For layer _i_, _#
    filters = INIT_FILTERS * (FILTER_EXP_BASE) <sup>i</sup>_
  - **CONV_BLOCKS**: The number of convolutional blocks. Each block
    contains a 2D convolutional layer, a batch normalization layer,
    activation layer, and a maxpool layer.
  - **NODES_DENSE0**: The number of nodes in the fully connected layer
    following flattening of parameters
  - **LR**: Learning rate
  - **DROPOUT**: Dropout rate
  - **L2_LAMBDA**: L2 regularization parameter
- **DCNN_MULTICLASS**: Contains definitions of configurable
  hyperparameters associated with a custom deep convolutional neural
  network for multi-class classification. The fields are identical to
  those in the _DCNN_BINARY_ subsection.
#### LIME
- **KERNEL_WIDTH**: Affects size of neighbourhood around which LIME
  samples for a particular example. In our experience, setting this
  within the continuous range of _[1.5, 2.0]_ is large enough to produce
  stable explanations, but small enough to avoid producing explanations
  that approach a global surrogate model.
- **FEATURE_SELECTION**: The strategy to select features for LIME
  explanations. Read the LIME creators'
  [documentation](https://lime-ml.readthedocs.io/en/latest/lime.html)
  for more information.
- **NUM_FEATURES**: The number of features to
  include in a LIME explanation
- **NUM_SAMPLES**: The number of samples used to fit a linear model when
  explaining a prediction using LIME
- **COVID_ONLY**: Set to _'true'_ if you want explanations to be
  provided for the predicted logit corresponding to the "COVID-19"
  class, despite the model's prediction. If set to _'false'_,
  explanations will be provided for the logit corresponding to the
  predicted class.
#### PREDICTION
- **THRESHOLD**: Classification threshold for prediction

## Contact
**Matt Ross**  
Manager, Artificial Intelligence  
Information Technology Services, City Manager’s Office  
City of London  
201 Queens Ave. Suite 300, London, ON. N6A 1J1  
P: 519.661.CITY (2489) x 5451 | C: 226.448.9113
