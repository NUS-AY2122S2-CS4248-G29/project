import numpy as np
import tensorflow as tf

from data import Dataset
from models import TensorFlowModel
from models.preprocessing import NltkTokenStopLemmaTfidf

class CsrMatrixTensorFlowModel(TensorFlowModel):
    def __init__(self):
        super().__init__()
        self._set_preprocessor(NltkTokenStopLemmaTfidf())

    def set_data(self, raw_dataset: Dataset) -> None:
        self._set_training_data(*raw_dataset.get_training_data())
        val_data = raw_dataset.get_validation_data()
        if val_data is not None:
            self._set_validation_data(*val_data)
        self._set_test_data(*raw_dataset.get_test_data())

    def _convert_to_tensorflow_dataset(self, raw_text_arr: np.ndarray, raw_label_arr: np.ndarray) -> tf.data.Dataset:
        preprocessor = self._get_preprocessor()
        text_sparse_matrix = preprocessor(raw_text_arr)
        text_sparse_matrix = text_sparse_matrix.tocoo()
        indices = np.mat([text_sparse_matrix.row, text_sparse_matrix.col]).transpose()
        text_tensor = tf.sparse.reorder(tf.SparseTensor(indices, text_sparse_matrix.data, text_sparse_matrix.shape))
        label_tensor = tf.one_hot(raw_label_arr, 4)
        ds = tf.data.Dataset.from_tensor_slices((text_tensor, label_tensor))
        ds = ds.map(lambda t, l: (tf.sparse.to_dense(t), l))
        return ds
