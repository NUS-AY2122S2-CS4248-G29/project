import tensorflow as tf

from models import NumpyTensorFlowModel
from models.tensorflow_models.layers import EnhancedMLP
from models.tensorflow_models.layers import EnhancedStackedRNN
from models.tensorflow_models.metrics.f1_score import F1Score

NUM_EPOCHS = 128

EMBEDDING_DIM = 64
NUM_LSTM_UNITS = 32
NUM_HIDDEN_NODES = 512
NUM_OUTPUT_NODES = 4

class ResidualLstmModel(NumpyTensorFlowModel):
    def __init__(self):
        super().__init__()

    def display_metrics(self) -> None:
        loss, accuracy, *f1 = self._get_metrics()
        print('Loss:', loss)
        print('Accuracy:', accuracy)
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

    def _get_number_of_epochs(self) -> int:
        return NUM_EPOCHS

class Model(tf.keras.Model):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, EMBEDDING_DIM,
            # embeddings_regularizer=tf.keras.regularizers.L2()
        )
        self.rnn = EnhancedStackedRNN(
            lambda num_nodes, *lc_args, **lc_kwargs: tf.keras.layers.LSTM(
                num_nodes,
                *lc_args,
                **lc_kwargs
            ),
            NUM_LSTM_UNITS,
            num_hidden_layers=1,
            has_residual=True,
            has_batch_norm=True,
            is_bidirectional=True
        )
        self.mid = EnhancedMLP(
            NUM_HIDDEN_NODES,
            activation=None,
            has_batch_norm=True
        )
        self.mlp = EnhancedMLP(
            NUM_OUTPUT_NODES,
            num_hidden_layers=2,
            num_hidden_nodes=NUM_HIDDEN_NODES,
            output_activation='softmax',
            dropout_rate=0.5,
            has_residual=True,
            has_batch_norm=True,
            has_output_residual=False,
            has_output_batch_norm=False
        )

    def call(self, x):
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.mid(x)
        x = self.mlp(x)
        return x
