import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras

tfd = tfp.distributions


def kl_divergence_function(q, p, _):
    return tfp.distributions.kl_divergence(q, p) / tf.cast(60000, dtype=tf.float32)


def block(filters: int, max_pool: bool = True):
    conv_layer = tfp.layers.Convolution2DFlipout(
        filters,
        kernel_size=5,
        padding="same",
        kernel_divergence_fn=kl_divergence_function,
        activation=tf.nn.relu,
    )
    if not max_pool:
        return (conv_layer,)
    max_pool = tf_keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="same")
    return conv_layer, max_pool


def get_model():
    model = tf_keras.models.Sequential(
        [
            *block(5),
            *block(16),
            *block(120, max_pool=False),
            tf_keras.layers.Flatten(),
            tfp.layers.DenseFlipout(
                84,
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu,
            ),
            tfp.layers.DenseFlipout(
                10,
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.softmax,
            ),
        ]
    )
    return model
