import tensorflow as tf

from models.tensorflow_models.layers import EnhancedRNN

class EnhancedStackedRNN(tf.keras.layers.Layer):
    def __init__(self,
        layer_creator,
        num_nodes,
        *args,
        num_hidden_layers=0,
        num_hidden_nodes=None,
        output_layer_creator=None,
        dropout_rate=0.0,
        has_residual=False,
        has_batch_norm=False,
        has_output_batch_norm=None,
        is_bidirectional=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if num_hidden_nodes is None:
            num_hidden_nodes = num_nodes
        self.layers = [EnhancedRNN(
            lambda num_nodes, *lc_args, **lc_kwargs: layer_creator(
                num_nodes,
                *lc_args,
                return_sequences=True,
                **lc_kwargs
            ),
            num_hidden_nodes,
            *args,
            dropout_rate=dropout_rate,
            has_residual=has_residual,
            has_batch_norm=has_batch_norm,
            is_bidirectional=is_bidirectional,
            **kwargs
        ) for _ in range(num_hidden_layers)]
        if output_layer_creator is None:
            output_layer_creator = layer_creator
        if has_output_batch_norm is None:
            has_output_batch_norm = has_batch_norm
        self.layers.append(EnhancedRNN(
            lambda num_nodes, *olc_args, **olc_kwargs: layer_creator(
                num_nodes,
                *olc_args,
                **olc_kwargs
            ),
            num_nodes,
            *args,
            dropout_rate=dropout_rate,
            has_batch_norm=has_output_batch_norm,
            is_bidirectional=is_bidirectional,
            **kwargs
        ))

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
