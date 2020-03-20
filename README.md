# COVID-19 Chest X-Ray Model
![alt text](documents/readme_images/london_logo.png "City of London logo")

## Project Structure
The project looks similar to the directory structure below. Disregard
any _.gitkeep_ files, as their only purpose is to force Git to track
empty directories. Disregard any _.\__init\__.py_ files, as they are
empty files that enable Python to recognize certain directories as
packages.

```
├── data
│   ├── interpretability          <- Generated feature information
│   ├── processed                 <- Products of preprocessing
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
- **RAW_OTHER_DATA**: Path to folder containing [Kaggle chest X-ray
  pneumonia dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **MODEL_WEIGHTS**: Path at which to save trained model's weights
- **MODEL_TO_LOAD**: Path to the trained model's weights that you would
  like to load for prediction
#### DATA
- **IMG_DIM**: Desired target size of image after preprocessing
- **VAL_SPLIT**: Fraction of the data allocated to the validation set
- **TEST_SPLIT**: Fraction of the data allocated to the test set
- **KAGGLE_DATA_FRAC**: Fraction of the images from the Kaggle chest
  X-ray dataset to use. The default value results in a dataset of > 1000
  images.
#### NN
- **DCNN1**: Contains definitions of configurable hyperparameters
  associated with a custom deep convolutional neural network. The values
  currently in this section were the optimal values for our dataset
  informed by heuristically selecting hyperparameters.
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

#### TRAIN
- **EXPERIMENT_TYPE**: The type of training experiment you would like to
  perform if executing [_train.py_](src/train.py). For now, the only
  choice is _'single_train'_.
- **EPOCHS**: Number of epochs to train the model for
- **THRESHOLDS**: A single float or list of floats in range [0, 1]
  defining the classification threshold. Affects precision and recall
  metrics.
- **BATCH_SIZE**: Mini-batch size during training
- **IMB_STRATEGY**: Class imbalancing strategy to employ. In our
  dataset, the ratio of positive to negative ground truth was very low,
  prompting the use of these strategies. Set either to _'class_weight'_
  or _'random_oversample'_.
- **POS_WEIGHT**: Coefficient to multiply the positive class' weight by
  during computation of loss function. Negative class' weight is
  multiplied by _(1 - POS_WEIGHT)_. Increasing this number tends to
  increase recall and decrease precision.

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
- **NUM_SAMPLES**: The number of samples
  used to fit a linear model when explaining a prediction using LIME

#### PREDICTION
- **THRESHOLD**: Classification threshold for prediction

## Contact
**Matt Ross**  
Manager, Artificial Intelligence  
Information Technology Services, City Manager’s Office  
City of London  
201 Queens Ave. Suite 300, London, ON. N6A 1J1  
P: 519.661.CITY (2489) x 5451 | C: 226.448.9113