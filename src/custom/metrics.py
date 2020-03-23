import numpy as np
from tensorflow.keras.metrics import Metric, Precision, Recall
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras import backend as K

class F1Score(Metric):
    '''
    Custom tf.keras metric that calculates the F1 Score
    '''

    def __init__(self, thresholds=None, top_k=None, class_id=None, name=None, dtype=None):
        '''
        Creates an instance of the  F1Score class
        :param thresholds: A float value or a python list/tuple of float threshold values in [0, 1].
        :param top_k: An int value specifying the top-k predictions to consider when calculating precision
        :param class_id: Integer class ID for which we want binary metrics. This must be in the half-open interval
                `[0, num_classes)`, where `num_classes` is the last dimension of predictions
        :param name: string name of the metric instance
        :param dtype: data type of the metric result
        '''
        super(F1Score, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.top_k = top_k
        self.class_id = class_id

        default_threshold = 0.5 if top_k is None else metrics_utils.NEG_INF
        self.thresholds = metrics_utils.parse_init_thresholds(
            thresholds, default_threshold=default_threshold)
        self.true_positives = self.add_weight('true_positives', shape=(len(self.thresholds),),
                                              initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight('false_positives', shape=(len(self.thresholds),),
                                               initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight('false_negatives', shape=(len(self.thresholds),),
                                               initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        '''
        Accumulates true positive, false positive and false negative statistics.
        :param y_true: The ground truth values, with the same dimensions as `y_pred`. Will be cast to `bool`
        :param y_pred: The predicted values. Each element must be in the range `[0, 1]`
        :param sample_weight: Weighting of each example. Defaults to 1. Can be a `Tensor` whose rank is either 0,
               or the same rank as `y_true`, and must be broadcastable to `y_true`
        :return: Update operation
        '''
        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true, y_pred, thresholds=self.thresholds, top_k=self.top_k, class_id=self.class_id,
            sample_weight=sample_weight)


    def result(self):
        '''
        Compute the value for the F1 score. Calculates precision and recall, then F1 score.
        F1 = 2 * precision * recall / (precision + recall)
        :return: F1 score
        '''
        precision = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_positives)
        recall = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        result = math_ops.div_no_nan(2 * precision * recall, precision + recall)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        '''
        Resets all of the metric state variables. Called between epochs, when a metric is evaluated during training.
        '''
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value(
            [(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        '''
        Returns the serializable config of the metric.
        :return: serializable config of the metric
        '''
        config = {
            'thresholds': self.init_thresholds,
            'top_k': self.top_k,
            'class_id': self.class_id
        }
        base_config = super(F1Score, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))