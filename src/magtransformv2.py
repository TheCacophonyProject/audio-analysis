import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="MyLayers", name="MagTransform")
class MagTransform(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MagTransform, self).__init__(**kwargs)
        self.a = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=-1.0),
            name="a-power",
            dtype="float32",
            shape=[1],
            trainable=True,
            constraint=tf.keras.constraints.MinMaxNorm(
                min_value=-2.0, max_value=1.0, rate=1.0, axis=-1
            ),
        )

    def call(self, inputs):
        c = tf.math.pow(inputs, tf.math.sigmoid(self.a))
        return c
