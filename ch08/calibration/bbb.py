import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def cnn_building_block_bbb(num_filters, kl_divergence_function):
    return tf.keras.Sequential(
        [
            tfp.layers.Convolution2DReparameterization(
                num_filters,
                kernel_size=(3, 3),
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu,
            ),
            tf.keras.layers.MaxPool2D(strides=2),
        ]
    )


def build_and_compile_model_bbb(num_train_examples: int):
    kl_divergence_function = lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(
        num_train_examples, dtype=tf.float32
    )

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=(32, 32, 3)),
            cnn_building_block_bbb(16, kl_divergence_function),
            cnn_building_block_bbb(32, kl_divergence_function),
            cnn_building_block_bbb(64, kl_divergence_function),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseReparameterization(
                64,
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu,
            ),
            tfp.layers.DenseReparameterization(
                10,
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.softmax,
            ),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )

    model.build(input_shape=[None, 32, 32, 3])
    return model


def get_bbb_predictions(bbb_model, images, num_inferences):
    bbb_predictions = tf.stack(
        [bbb_model.predict(images) for _ in range(num_inferences)],
        axis=0,
    )
    return np.mean(bbb_predictions, axis=0)
