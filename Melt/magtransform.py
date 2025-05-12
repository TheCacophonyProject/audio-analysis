import tensorflow as tf


# tensorflow refuces to load without this
@tf.keras.utils.register_keras_serializable(package="MyLayers", name="MagTransform")
class MagTransform(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MagTransform, self).__init__(**kwargs)
        self.a = self.add_weight(
            initializer=tf.keras.initializers.Constant(value=0.0),
            name="a-power",
            dtype="float32",
            shape=(),
            trainable=True,
        )

    def call(self, inputs):
        c = tf.math.pow(inputs, tf.math.sigmoid(self.a))
        return c
