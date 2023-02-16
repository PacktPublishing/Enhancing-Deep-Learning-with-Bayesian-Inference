import tensorflow as tf


def cnn_building_block(num_filters):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPool2D(strides=2),
        ]
    )


def build_and_compile_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=(32, 32, 3)),
            cnn_building_block(16),
            cnn_building_block(32),
            cnn_building_block(64),
            tf.keras.layers.MaxPool2D(strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
