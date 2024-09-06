import tensorflow as tf
from tensorflow.keras import Model


def conv_block(filters):
    return [
        tf.keras.layers.Conv2D(
            filters,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
    ]


def get_model() -> Model:
   return tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            input_shape=(160, 160, 3),
            kernel_initializer="he_uniform",
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        *conv_block(64),
        *conv_block(128),
        *conv_block(256),
        *conv_block(128),
        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_uniform",
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2),
    ]
)
