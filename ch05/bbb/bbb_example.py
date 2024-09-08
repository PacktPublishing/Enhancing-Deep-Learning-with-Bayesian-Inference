#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of Bayes By Backprop."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

NUM_INFERENCES = 7


# download mnist fashion data set
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# set clas names
CLASS_NAMES = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# derive number training examples and number of classes
NUM_TRAIN_EXAMPLES = len(train_images)
NUM_CLASSES = len(CLASS_NAMES)


def define_bayesian_model():
    """Define the architecture of the Bayesian network."""

    def kl_divergence_function(q, p, _):
        return tfp.distributions.kl_divergence(q, p) / tf.cast(NUM_TRAIN_EXAMPLES, dtype=tf.float32)

    model = tf.keras.models.Sequential(
        [
            tfp.layers.Convolution2DReparameterization(
                64,
                kernel_size=5,
                padding="SAME",
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu,
            ),
            tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=[2, 2], padding="SAME"),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseReparameterization(
                NUM_CLASSES, kernel_divergence_fn=kl_divergence_function, activation=tf.nn.softmax
            ),
        ]
    )
    return model


def compile_bayesian_model(model):
    """Compile the Bayesian network."""
    # define the optimizer
    optimizer = tf.keras.optimizers.Adam()
    # compile the model
    model.compile(
        optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )
    # build the model
    model.build(input_shape=[None, 28, 28, 1])
    return model


# convert training labels from integer to one-hot vectors
train_labels_dense = tf.one_hot(train_labels, NUM_CLASSES)
# use helper function to define the model architecture
bayesian_model = define_bayesian_model()
# use helper function to compile the model
bayesian_model = compile_bayesian_model(bayesian_model)
# initiate model training
bayesian_model.fit(train_images, train_labels_dense, epochs=10)

# create predictions with the trained model.
NUM_SAMPLES_INFERENCE = 5
softmax_predictions = tf.stack(
    [bayesian_model.predict(test_images[:NUM_SAMPLES_INFERENCE]) for _ in range(NUM_INFERENCES)],
    axis=0,
)

# get the class predictions for the first image in the test set
image_ind = 0
# collect class predictions
class_predictions = []
for ind in range(NUM_INFERENCES):
    prediction_this_inference = np.argmax(softmax_predictions[ind][image_ind])
    class_predictions.append(prediction_this_inference)
# get class predictions in human-readable form
predicted_classes = [CLASS_NAMES[ind] for ind in class_predictions]

# define image caption
image_caption = (
    f"Sample 1: {predicted_classes[0]}\n"
    + f"Sample 2: {predicted_classes[1]}\n"
    + f"Sample 3: {predicted_classes[2]}\n"
    + f"Sample 4: {predicted_classes[3]}\n"
    + f"Sample 5: {predicted_classes[4]}\n"
    + f"Sample 6: {predicted_classes[5]}\n"
    + f"Sample 7: {predicted_classes[6]}\n"
)
# visualise image and predictions
plt.figure(dpi=300)
plt.title(f"Correct class: {CLASS_NAMES[test_labels[image_ind]]}")
plt.imshow(test_images[image_ind], cmap=plt.cm.binary)
plt.xlabel(image_caption)
plt.show()
