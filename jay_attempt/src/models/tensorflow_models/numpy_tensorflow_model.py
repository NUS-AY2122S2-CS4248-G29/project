import numpy as np
import tensorflow as tf

from data import Dataset
from models import TensorFlowModel
from models.preprocessing import BasicSequence

class NumpyTensorFlowModel(TensorFlowModel):
    def __init__(self):
        super().__init__()
        self._set_preprocessor(BasicSequence())

    def set_data(self, raw_dataset: Dataset) -> None:
        self._set_training_data(*raw_dataset.get_training_data())
        val_data = raw_dataset.get_validation_data()
        if val_data is not None:
            self._set_validation_data(*val_data)
        self._set_test_data(*raw_dataset.get_test_data())

    def _convert_to_tensorflow_dataset(self, raw_text_arr: np.ndarray, raw_label_arr: np.ndarray) -> tf.data.Dataset:
        preprocessor = self._get_preprocessor()
        text_arr = preprocessor(raw_text_arr)
        text_tensor = tf.ragged.constant(text_arr)
        label_tensor = tf.one_hot(raw_label_arr, 4)
        ds = tf.data.Dataset.from_tensor_slices((text_tensor, label_tensor))
        ds = ds.map(lambda t, l: (tf.convert_to_tensor(t), l))
        return ds
