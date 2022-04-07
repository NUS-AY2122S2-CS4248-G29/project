import tensorflow as tf

class EnhancedRNN(tf.keras.layers.Layer):
    def __init__(self,
        layer_creator,
        num_nodes,
        *args,
        dropout_rate=0.0,
        has_residual=False,
        has_batch_norm=False,
        is_bidirectional=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.layer = layer_creator(num_nodes, *args, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.has_residual = has_residual
        self.has_batch_norm = has_batch_norm
        if has_batch_norm:
            axes = [-1]
            if 'return_sequences' in kwargs:
                axes.append(-2)
            self.batch_norm = tf.keras.layers.BatchNormalization(axis=axes)
        if is_bidirectional:
            self.layer = tf.keras.layers.Bidirectional(self.layer)

    def call(self, x):
        x = self.dropout(x)
        r = x
        x = self.layer(x)
        if self.has_residual:
            x += r
        if self.has_batch_norm:
            x = self.batch_norm(x)
        return x
