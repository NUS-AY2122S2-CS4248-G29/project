from typing import List
from typing import Union

import numpy as np
import tensorflow as tf

from data import Dataset
from models import Model
from models.preprocessing import Preprocessor
from models.preprocessing import BasicSequence

SHUFFLE_BUFFER_SIZE = 65536
MAX_BATCH_SIZE = 1024
DATASET_NUM_BUCKETS = 18
DATASET_BUCKET_BOUNDARIES = [2 ** i for i in range(DATASET_NUM_BUCKETS)]
DATASET_BUCKET_BATCH_SIZES = [min(2 ** i, MAX_BATCH_SIZE) for i in range(DATASET_NUM_BUCKETS)][::-1] + [1]
DEFAULT_NUMBER_OF_EPOCHS = 10
TRAIN_VAL_SPLIT_FOLD = 5

class TensorFlowModel(Model):
    def __init__(self):
        super().__init__()
        self._set_preprocessor(BasicSequence())

    def set_data(self, raw_dataset: Dataset) -> None:
        self._set_training_data(*raw_dataset.get_training_data())
        val_data = raw_dataset.get_validation_data()
        if val_data is not None:
            self._set_validation_data(*val_data)
        self._set_test_data(*raw_dataset.get_test_data())

    def train(self) -> None:
        model = self._get_tensorflow_model()
        model.summary()
        train_ds = self._get_training_dataset()
        train_ds = self._finalise_dataset(
            train_ds,
            shuffle_buffer_size=SHUFFLE_BUFFER_SIZE
        )
        num_epochs = self._get_number_of_epochs()
        callbacks = self._get_callbacks()
        val_ds = self._get_validation_dataset()
        if val_ds is not None:
            val_ds = self._finalise_dataset(val_ds)
        history = model.fit(
            train_ds,
            epochs=num_epochs,
            callbacks=callbacks,
            validation_data=val_ds
        )

    def evaluate(self) -> None:
        model = self._get_tensorflow_model()
        test_ds = self._get_test_dataset()
        test_ds = self._finalise_dataset(test_ds)
        metrics = model.evaluate(test_ds, return_dict=True)
        self._set_metrics(metrics)

    def _create_tensorflow_model(self) -> tf.keras.Model:
        pass

    def _convert_to_tensorflow_dataset(self, raw_text_arr: np.ndarray, raw_label_arr: np.ndarray) -> tf.data.Dataset:
        pass

    def _set_training_data(self, raw_text_arr: np.ndarray, raw_label_arr: np.ndarray) -> None:
        preprocessor = self._get_preprocessor()
        preprocessor.adapt(raw_text_arr)
        train_ds = self._convert_to_tensorflow_dataset(raw_text_arr, raw_label_arr)
        self._set_training_dataset(train_ds)

    def _set_validation_data(self, raw_text_arr: np.ndarray, raw_label_arr: np.ndarray) -> None:
        val_ds = self._convert_to_tensorflow_dataset(raw_text_arr, raw_label_arr)
        self._set_validation_dataset(val_ds)

    def _set_test_data(self, raw_text_arr: np.ndarray, raw_label_arr: np.ndarray) -> None:
        test_ds = self._convert_to_tensorflow_dataset(raw_text_arr, raw_label_arr)
        self._set_test_dataset(test_ds)

    def _get_tensorflow_model(self) -> tf.keras.Model:
        model = None
        try:
            model = self._tensorflow_model
        except AttributeError:
            self._tensorflow_model = self._create_tensorflow_model()
            model = self._tensorflow_model
        load_filepath = self.get_load_filepath()
        if load_filepath is not None:
            model.load_weights(load_filepath).expect_partial()
        return model

    def _batch_dataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.bucket_by_sequence_length(
            element_length_func=lambda text, label: tf.shape(text)[0],
            bucket_boundaries=DATASET_BUCKET_BOUNDARIES,
            bucket_batch_sizes=DATASET_BUCKET_BATCH_SIZES
        )
        return dataset

    def _finalise_dataset(self,
        dataset: tf.data.Dataset,
        shuffle_buffer_size: Union[int, None] = None,
        must_batch: bool = True,
        prefetch_buffer_size: Union[int, None] = tf.data.AUTOTUNE
    ) -> tf.data.Dataset:
        if shuffle_buffer_size is not None:
            dataset = dataset.shuffle(shuffle_buffer_size)
        if must_batch:
            dataset = self._batch_dataset(dataset)
        if prefetch_buffer_size is not None:
            dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset

    def _set_preprocessor(self, preprocessor: Preprocessor) -> None:
        self._preprocessor = preprocessor

    def _get_preprocessor(self) -> Preprocessor:
        return self._preprocessor

    def _set_training_dataset(self, training_dataset: tf.data.Dataset) -> None:
        self._train_ds = training_dataset

    def _get_training_dataset(self) -> tf.data.Dataset:
        return self._train_ds

    def _set_validation_dataset(self, validation_dataset: tf.data.Dataset) -> None:
        self._val_ds = validation_dataset

    def _get_validation_dataset(self) -> Union[tf.data.Dataset, None]:
        try:
            return self._val_ds
        except AttributeError:
            return None

    def _set_test_dataset(self, test_dataset: tf.data.Dataset) -> None:
        self._test_ds = test_dataset

    def _get_test_dataset(self) -> tf.data.Dataset:
        return self._test_ds

    def _get_number_of_epochs(self) -> int:
        return DEFAULT_NUMBER_OF_EPOCHS

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=1e-4,
                patience=5,
                mode='min',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                patience=2,
                min_delta=1e-4,
                mode='min'
            )
        ]
        save_filepath = self.get_save_filepath()
        if save_filepath:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=save_filepath,
                save_weights_only=True,
                verbose=1
            )
            callbacks.append(cp_callback)
        return callbacks

    def _set_metrics(self, metrics) -> None:
        self._metrics = metrics

    def _get_metrics(self):
        return self._metrics
