import ddu_dirty_mnist
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


def get_data():
    dirty_mnist_train = ddu_dirty_mnist.DirtyMNIST(
        ".", train=True, download=True, normalize=False, noise_stddev=0
    )
    # regular MNIST
    train_imgs = dirty_mnist_train.datasets[0].data.numpy()
    train_labels = dirty_mnist_train.datasets[0].targets.numpy()
    # AmbiguousMNIST
    train_imgs_amb = dirty_mnist_train.datasets[1].data.numpy()
    train_labels_amb = dirty_mnist_train.datasets[1].targets.numpy()

    train_imgs, train_labels = shuffle(
        np.concatenate([train_imgs, train_imgs_amb]),
        np.concatenate([train_labels, train_labels_amb]),
    )
    train_imgs = np.expand_dims(train_imgs[:, 0, :, :], -1)
    train_labels = tf.one_hot(train_labels, 10)

    (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data()[1]
    test_imgs = test_imgs / 255.0
    test_imgs = np.expand_dims(test_imgs, -1)
    test_labels = tf.one_hot(test_labels, 10)
    return train_imgs, train_labels, test_imgs, test_labels
