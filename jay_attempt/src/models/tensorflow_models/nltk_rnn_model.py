from typing import List

import tensorflow as tf

from models import NumpyTensorFlowModel
from models.preprocessing import NltkTokenStopLemmaSequence
from models.tensorflow_models.layers import EnhancedMLP
from models.tensorflow_models.layers import EnhancedStackedRNN
from models.tensorflow_models.metrics import F1Score

NUM_EPOCHS = 128

EMBEDDING_DIM = 4
NUM_RNN_UNITS = 4
NUM_OUTPUT_NODES = 4

class NltkRnnModel(NumpyTensorFlowModel):
    def __init__(self):
        super().__init__()
        self._set_preprocessor(NltkTokenStopLemmaSequence())

    def display_metrics(self) -> None:
        loss, accuracy, f1_macro, f1_micro, *f1 = self._get_metrics()
        print('Loss:', loss)
        print('Accuracy:', accuracy)
        print('Macro F1 score:', f1_macro)
        print('Micro F1 score:', f1_micro)
        for i in range(len(f1)):
            print(f'Class {i + 1} F1 score: {f1[i]}')

    def _create_tensorflow_model(self) -> tf.keras.Model:
        preprocessor = self._get_preprocessor()
        vocab_size = preprocessor.get_vocabulary_size()
        model = Model(vocab_size)
        model.build(tf.TensorShape([None, None]))
        initial_learning_rate = 1e-2
        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            metrics=['accuracy', F1Score(4)]
        )
        return model

    def _get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        callbacks = super()._get_callbacks()
        callbacks.extend([
            tf.keras.callbacks.EarlyStopping(
                monitor='f1_macro',
                min_delta=1e-4,
                patience=5,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='f1_micro',
                min_delta=1e-4,
                patience=5,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='f1_1',
                min_delta=1e-4,
                patience=5,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='f1_2',
                min_delta=1e-4,
                patience=5,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='f1_3',
                min_delta=1e-4,
                patience=5,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='f1_4',
                min_delta=1e-4,
                patience=5,
                mode='max',
                restore_best_weights=True
            )
        ])
        return callbacks

    def _get_number_of_epochs(self) -> int:
        return NUM_EPOCHS

class Model(tf.keras.Model):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            EMBEDDING_DIM
        )
        self.rnn = EnhancedStackedRNN(
            lambda num_nodes, *lc_args, **lc_kwargs: tf.keras.layers.LSTM(
                num_nodes,
                *lc_args,
                # kernel_regularizer=tf.keras.regularizers.L2(),
                # recurrent_regularizer=tf.keras.regularizers.L2(),
                # bias_regularizer=tf.keras.regularizers.L2(),
                **lc_kwargs
            ),
            NUM_RNN_UNITS,
            has_batch_norm=True,
            is_bidirectional=True
        )
        self.mlp = EnhancedMLP(
            NUM_OUTPUT_NODES,
            output_activation='softmax',
            # dropout_rate=0.5,
            # kernel_regularizer=tf.keras.regularizers.L2(),
            # bias_regularizer=tf.keras.regularizers.L2()
        )

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.mlp(x)
        return x
