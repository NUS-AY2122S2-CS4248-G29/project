from typing import Dict

import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='f1_score', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self._num_classes = num_classes
        self._confusion_matrix = self.add_weight(
            name='confusion_matrix',
            shape=(num_classes, num_classes),
            dtype=tf.float32,
            initializer='zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None) -> None:
        labels = tf.math.argmax(y_true, -1)
        predictions = tf.math.argmax(y_pred, -1)
        partial_confusion_matrix = tf.math.confusion_matrix(labels, predictions, num_classes=self._num_classes)
        partial_confusion_matrix  = tf.cast(partial_confusion_matrix, tf.float32)
        self._confusion_matrix.assign_add(partial_confusion_matrix)

    def result(self) -> Dict[int, tf.Tensor]:
        predicted_positives = tf.math.reduce_sum(self._confusion_matrix, axis=0)
        actual_positives = tf.math.reduce_sum(self._confusion_matrix, axis=1)
        true_positives = tf.linalg.tensor_diag_part(self._confusion_matrix)
        precisions = tf.math.divide_no_nan(true_positives, predicted_positives)
        recalls = tf.math.divide_no_nan(true_positives, actual_positives)
        f1_scores = 2 * tf.math.divide_no_nan(tf.math.multiply(precisions, recalls), precisions + recalls)
        results = dict()
        results['f1_macro'] = tf.reduce_mean(f1_scores)
        false_positives = predicted_positives - true_positives
        false_negatives = actual_positives - true_positives
        true_positives_sum = tf.reduce_sum(true_positives)
        false_positives_sum = tf.reduce_sum(false_positives)
        false_negatives_sum = tf.reduce_sum(false_negatives)
        results['f1_micro'] = tf.math.divide_no_nan(true_positives_sum, true_positives_sum + 0.5 * (false_positives_sum + false_negatives_sum))
        results = {
            **results,
            **{f'f1_{i + 1}': f1_scores[i] for i in range(len(f1_scores))}
        }
        return results

    def reset_state(self) -> None:
        self._confusion_matrix.assign(tf.zeros_like(self._confusion_matrix))
