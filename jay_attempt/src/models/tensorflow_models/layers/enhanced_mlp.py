import tensorflow as tf

from models.tensorflow_models.layers import EnhancedDense

class EnhancedMLP(tf.keras.layers.Layer):
    def __init__(self,
        num_nodes,
        *args,
        num_hidden_layers=0,
        num_hidden_nodes=None,
        kernel_regularizer=None,
        bias_regularizer=None,
        activation='relu',
        output_activation=None,
        dropout_rate=0.0,
        has_residual=False,
        has_batch_norm=False,
        has_output_residual=None,
        has_output_batch_norm=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if num_hidden_nodes is None:
            num_hidden_nodes = num_nodes
        self.layers = [EnhancedDense(
            num_hidden_nodes,
            *args,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dropout_rate=dropout_rate,
            has_residual=has_residual,
            has_batch_norm=has_batch_norm,
            **kwargs
        ) for _ in range(num_hidden_layers)]
        if output_activation is None:
            output_activation = activation
        if has_output_residual is None:
            has_output_residual = has_residual
        if has_output_batch_norm is None:
            has_output_batch_norm = has_batch_norm
        self.layers.append(EnhancedDense(
            num_nodes,
            *args,
            activation=output_activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            dropout_rate=dropout_rate,
            has_residual=has_output_residual,
            has_batch_norm=has_output_batch_norm,
            **kwargs
        ))

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
