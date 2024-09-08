import dataclasses
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


@dataclasses.dataclass
class Data:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_train_al: Optional[np.ndarray] = None
    y_train_al: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        repr_str = ""
        for field in dataclasses.fields(self):
            repr_str += f"{field.name}: {getattr(self, field.name).shape} \n"
        return repr_str


def get_data() -> Data:
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    return Data(x_train, y_train, x_test, y_test)


def get_random_balanced_indices(data: Data, initial_n_samples: int) -> np.ndarray:
    labels = np.argmax(data.y_train, axis=1)
    indices = []
    label_list = np.unique(labels)
    for label in label_list:
        indices_label = np.random.choice(
            np.argwhere(labels == label).flatten(),
            size=initial_n_samples // len(label_list),
            replace=False,
        )
        indices.extend(indices_label)
    indices = np.array(indices)
    np.random.shuffle(indices)
    return indices


def get_initial_ds(data: Data, initial_n_samples: int) -> Data:
    indices = get_random_balanced_indices(data, initial_n_samples)
    x_train_al, y_train_al = data.x_train[indices], data.y_train[indices]
    x_train = np.delete(data.x_train, indices, axis=0)
    y_train = np.delete(data.y_train, indices, axis=0)
    return Data(x_train, y_train, data.x_test, data.y_test, x_train_al, y_train_al)


def update_ds(data: Data, indices_to_add: np.ndarray) -> Tuple[Data, Tuple[np.ndarray, np.ndarray]]:
    x_added, y_added = data.x_train[indices_to_add], data.y_train[indices_to_add]
    x_train_al = np.concatenate([data.x_train_al, x_added], axis=0)
    y_train_al = np.concatenate([data.y_train_al, y_added], axis=0)
    x_train = np.delete(data.x_train, indices_to_add, axis=0)
    y_train = np.delete(data.y_train, indices_to_add, axis=0)
    # shuffle set to train on
    x_train_al, y_train_al = shuffle(x_train_al, y_train_al, random_state=0)
    return Data(x_train, y_train, data.x_test, data.y_test, x_train_al, y_train_al), (
        x_added,
        y_added,
    )
