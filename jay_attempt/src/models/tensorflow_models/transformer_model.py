from typing import List

import tensorflow as tf

from models import NumpyTensorFlowModel
from models.preprocessing import NltkTokenStopLemmaSequence
from models.tensorflow_models.layers import EnhancedMLP
from models.tensorflow_models.metrics import F1Score

NUM_EPOCHS = 128
EARLY_STOPPING_PATIENCE = 10

EMBEDDING_DIM = 4
NUM_QUERY_KEY_NODES = 8
# NUM_VAL_NODES = 16
NUM_ATTENTION_HEADS = 4
NUM_OUTPUT_NODES = 4

class TransformerModel(NumpyTensorFlowModel):
    def __init__(self):
        super().__init__()

    def display_metrics(self) -> None:
        metrics = self._get_metrics()
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value}')

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
        callbacks = []
        callbacks.extend([
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss',
                patience=2,
                min_delta=1e-4,
                mode='min'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_macro',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_micro',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_1',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_2',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_3',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_4',
                min_delta=1e-4,
                patience=EARLY_STOPPING_PATIENCE,
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
        self.query = EnhancedMLP(
            NUM_QUERY_KEY_NODES,
            output_activation='relu',
            # kernel_regularizer=tf.keras.regularizers.L2(),
            # bias_regularizer=tf.keras.regularizers.L2()
        )
        self.query_pooling = tf.keras.layers.GlobalAveragePooling1D()
        # self.key = EnhancedMLP(
        #     NUM_QUERY_KEY_NODES,
        #     output_activation='relu',
        #     # kernel_regularizer=tf.keras.regularizers.L2(),
        #     # bias_regularizer=tf.keras.regularizers.L2()
        # )
        # self.value = EnhancedMLP(
        #     NUM_VAL_NODES,
        #     output_activation='relu',
        #     # kernel_regularizer=tf.keras.regularizers.L2(),
        #     # bias_regularizer=tf.keras.regularizers.L2()
        # )
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            NUM_ATTENTION_HEADS,
            NUM_QUERY_KEY_NODES,
            # output_shape=NUM_OUTPUT_NODES,
            # kernel_regularizer=tf.keras.regularizers.L2(),
            # bias_regularizer=tf.keras.regularizers.L2()
        )
        self.mlp = EnhancedMLP(
            NUM_OUTPUT_NODES,
            activation='softmax',
            # kernel_regularizer=tf.keras.regularizers.L2(),
            # bias_regularizer=tf.keras.regularizers.L2()
        )

    def call(self, x):
        x = self.embedding(x)
        q = self.query(x)
        q = self.query_pooling(q)
        # q = self.query_pooling(x)
        q = tf.expand_dims(q, 1)
        # k = self.key(x)
        # v = self.value(x)
        k = x
        v = x
        x = self.multi_head_attention(q, v, k)
        x = tf.squeeze(x, 1)
        x = self.mlp(x)
        return x
