import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import io
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from skimage.segmentation import mark_boundaries

# Set some matplotlib parameters
mpl.rcParams['figure.figsize'] = (12, 10)

def plot_to_tensor():
    '''
    Converts a matplotlib figure to an image tensor
    :param figure: A matplotlib figure
    :return: Tensorflow tensor representing the matplotlib image
    '''
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)     # Convert .png buffer to tensorflow image
    image = tf.expand_dims(image, 0)     # Add the batch dimension
    return image

def plot_metrics(history, metrics, dir_path=None):
    '''
    Plot metrics for the training and validation sets over the training history.
    :param history: Model history; returned from model.fit()
    :param metrics: List of metrics to plot
    :param dir_path: Directory in which to save image
    '''
    plt.clf()
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,3,n+1)
        plt.plot(history.epoch,  history.history[metric], label='Train')    # Plot metric on training data
        plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')    # Plot metric on validation data
        plt.xlabel('Epoch')
        plt.ylabel(name)

        # Set plot limits depending on the metric
        if metric == 'loss':
          plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
          plt.ylim([0.8,1])
        else:
          plt.ylim([0,1])
        plt.legend()
    if dir_path is not None:
        plt.savefig(dir_path + 'metrics_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return

def plot_roc(name, labels, predictions, class_id=0, dir_path=None):
    '''
    Plots the ROC curve for predictions on a dataset
    :param name: Name of dataset on the plot
    :param labels: Ground truth labels
    :param predictions: Model predictions corresponding to the labels
    :param dir_path: Directory in which to save image
    '''
    plt.clf()
    if np.max(labels) > 1:
        predictions = (np.argmax(predictions, axis=1) == labels)
        labels = (labels == class_id)
    fp, tp, _ = roc_curve(labels, predictions)  # Get values for true positive and true negative
    plt.plot(100*fp, 100*tp, label=name, linewidth=2)   # Plot the ROC curve
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    if dir_path is not None:
        plt.savefig(dir_path + 'ROC_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return plot_to_tensor()

def plot_confusion_matrix(labels, predictions, class_id=0, p=0.5, dir_path=None):
    '''
    Plot a confusion matrix for the ground truth labels and corresponding model predictions.
    :param labels: Ground truth labels
    :param predictions: Model predictions
    :param p: Classification threshold
    :param dir_path: Directory in which to save image
    '''
    plt.clf()
    if np.max(labels) > 1:
        predictions = (np.argmax(predictions, axis=1) == labels)
        labels = (labels == class_id)
    ax = plt.subplot()
    cm = confusion_matrix(labels, predictions > p)  # Calculate confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # Plot confusion matrix
    ax.figure.colorbar(im, ax=ax)
    ax.set(yticks=[-0.5, 1.5], xticks=[0, 1], yticklabels=['0', '1'], xticklabels=['0', '1'])
    ax.yaxis.set_major_locator(mpl.ticker.IndexLocator(base=1, offset=0.5))

    # Print number of TPs, FPs, TNs, FNs on each quadrant in the plot
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    # Set plot's title and axis names
    plt.title('Confusion matrix p={:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    # Save the image
    if dir_path is not None:
        plt.savefig(dir_path + 'CM_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')

    # Print these statistics
    print('True (-)ves: ', cm[0][0], '\nFalse (+)ves: ', cm[0][1], '\nFalse (-)ves: ', cm[1][0], '\nTrue (+)ves: ',
          cm[1][1])
    return plot_to_tensor()


def visualize_explanation(orig_img, explanation, img_filename, label, probs, file_path=None):
    '''
    Visualize an explanation for the prediction of a single X-ray image.
    :param orig_img: Original X-Ray image
    :param explanation: ImageExplanation object
    :param img_filename: Filename of the image explained
    :param label: Ground truth class of the example
    :param probs: Prediction probabilities
    :param file_path: Path to save the generated image
    '''

    # Plot original image on the left
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(orig_img)

    # Plot the image and its explanation on the right
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)
    ax[1].imshow(mark_boundaries(temp, mask))

    # Display some information about the example
    fig.text(0.02, 0.8, "Prediction probabilities: ['0': {:.2f}, '1': {:.2f}]".format(probs[0], probs[1]), fontsize=10)
    fig.text(0.02, 0.82, "Ground Truth: " + str(label), fontsize=10)
    fig.suptitle("LIME Explanation for image " + img_filename, fontsize=15)
    fig.tight_layout()

    # Save the image
    if file_path is not None:
        plt.savefig(file_path + img_filename + '_exp_' +
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.png')
    return