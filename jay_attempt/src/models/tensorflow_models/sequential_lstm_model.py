from typing import List

import tensorflow as tf

from models import NumpyTensorFlowModel

NUM_EPOCHS = 4

EMBEDDING_DIM = 32
NUM_LSTM_UNITS = 128
NUM_HIDDEN_NODES = 512
NUM_OUTPUT_NODES = 4

class SequentialLstmModel(NumpyTensorFlowModel):
    def __init__(self):
        super().__init__()

    def display_metrics(self) -> None:
        loss, accuracy = self._get_metrics()
        print('Loss:', loss)
        print('Accuracy:', accuracy)

    def _create_tensorflow_model(self) -> tf.keras.Model:
        preprocessor = self._get_preprocessor()
        vocab_size = preprocessor.get_vocabulary_size()
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                vocab_size, EMBEDDING_DIM,
                # embeddings_regularizer=tf.keras.regularizers.L2()
            ),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                NUM_LSTM_UNITS,
                # kernel_regularizer=tf.keras.regularizers.L2(),
                # recurrent_regularizer=tf.keras.regularizers.L2(),
                # bias_regularizer=tf.keras.regularizers.L2()
            )),
            # tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                NUM_HIDDEN_NODES, activation='relu',
                kernel_regularizer=tf.keras.regularizers.L2(),
                bias_regularizer=tf.keras.regularizers.L2()
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                NUM_HIDDEN_NODES, activation='relu',
                kernel_regularizer=tf.keras.regularizers.L2(),
                bias_regularizer=tf.keras.regularizers.L2()
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                NUM_OUTPUT_NODES, activation='softmax',
                # kernel_regularizer=tf.keras.regularizers.L2(),
                # bias_regularizer=tf.keras.regularizers.L2()
            )
        ])
        initial_learning_rate = 1e-2
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=7e2,
            decay_rate=0.1
        )
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=['accuracy']
        )
        return model

    def _get_number_of_epochs(self) -> int:
        return NUM_EPOCHS

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = []
        save_filepath = self.get_save_filepath()
        if save_filepath:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=save_filepath,
                save_weights_only=True,
                verbose=1
            )
            callbacks.append(cp_callback)
        return callbacks
