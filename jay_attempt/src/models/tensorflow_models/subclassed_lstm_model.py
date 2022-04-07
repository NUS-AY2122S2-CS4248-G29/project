from typing import List

import tensorflow as tf

from models import NumpyTensorFlowModel
from models.tensorflow_models.layers import EnhancedMLP
from models.tensorflow_models.layers import EnhancedStackedRNN

NUM_EPOCHS = 4

EMBEDDING_DIM = 32
NUM_LSTM_UNITS = 128
NUM_HIDDEN_NODES = 512
NUM_OUTPUT_NODES = 4

class SubclassedLstmModel(NumpyTensorFlowModel):
    def __init__(self):
        super().__init__()

    def display_metrics(self) -> None:
        loss, accuracy = self._get_metrics()
        print('Loss:', loss)
        print('Accuracy:', accuracy)

    def _create_tensorflow_model(self) -> tf.keras.Model:
        preprocessor = self._get_preprocessor()
        vocab_size = preprocessor.get_vocabulary_size()
        model = Model(vocab_size)
        model.build(tf.TensorShape([None, None]))
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

class Model(tf.keras.Model):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embedding = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)
        self._rnn = EnhancedStackedRNN(
            lambda num_nodes, *lc_args, **lc_kwargs: tf.keras.layers.LSTM(
                num_nodes,
                *lc_args,
                **lc_kwargs
            ),
            NUM_LSTM_UNITS,
            is_bidirectional=True
        )
        self._mlp1 = EnhancedMLP(NUM_HIDDEN_NODES,
            kernel_regularizer=tf.keras.regularizers.L2(),
            bias_regularizer=tf.keras.regularizers.L2()
        )
        self._mlp2 = EnhancedMLP(NUM_HIDDEN_NODES, dropout_rate=0.5,
            kernel_regularizer=tf.keras.regularizers.L2(),
            bias_regularizer=tf.keras.regularizers.L2()
        )
        self._mlp3 = EnhancedMLP(NUM_OUTPUT_NODES, output_activation='softmax', dropout_rate=0.5)

    def call(self, x):
        x = self._embedding(x)
        x = self._rnn(x)
        x = self._mlp1(x)
        x = self._mlp2(x)
        x = self._mlp3(x)
        return x
