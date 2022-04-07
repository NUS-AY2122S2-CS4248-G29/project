import tensorflow as tf

class EnhancedDense(tf.keras.layers.Dense):
    def __init__(self,
        *args,
        dropout_rate=0.0,
        has_residual=False,
        has_batch_norm=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.has_residual = has_residual
        self.has_batch_norm = has_batch_norm
        if has_batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, x):
        x = self.dropout(x)
        r = x
        x = super().call(x)
        if self.has_residual:
            x += r
        if self.has_batch_norm:
            x = self.batch_norm(x)
        return x
