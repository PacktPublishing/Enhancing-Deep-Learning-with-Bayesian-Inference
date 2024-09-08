"""
Data preprocessing and dataset creation for adversarial training.

This module provides functions to preprocess images and create datasets
for training, validation, and testing using the Oxford-IIIT Pet Dataset.
It includes utilities for image resizing, label encoding, and dataset
creation using TensorFlow's data API.

Functions:
    preprocess_image: Preprocess a single image file.
    preprocess: Preprocess an image file and its label.
    get_datasets: Create train and validation datasets.
    get_test_dataset: Create a test dataset.
    get_train_val_datasets: Create preprocessed train and validation datasets.

Constants:
    IMG_SIZE: Tuple defining the target size for resized images.
    AUTOTUNE: TensorFlow's autotune option for dataset performance.
"""

from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def preprocess_image(filename):
    """
    Preprocess an image file for the model.

    Args:
        filename (str): Path to the image file.

    Returns:
        tf.Tensor: Resized image tensor.
    """
    raw = tf.io.read_file(filename)
    image = tf.image.decode_png(raw, channels=3)
    return tf.image.resize(image, IMG_SIZE)


@tf.function
def preprocess(filename, label):
    """
    Preprocess an image file and its label.

    Args:
        filename (str): Path to the image file.
        label (int): Label of the image.

    Returns:
        tuple: Preprocessed image tensor and one-hot encoded label.
    """
    return preprocess_image(filename), tf.one_hot(label, 2)


def get_datasets(paths_train, labels_train, paths_val, labels_val):
    """
    Create train and validation datasets.

    Args:
        paths_train (list): List of paths to training images.
        labels_train (list): List of labels for training images.
        paths_val (list): List of paths to validation images.
        labels_val (list): List of labels for validation images.

    Returns:
        tuple: Train and validation tf.data.Dataset objects.
    """
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((paths_train, labels_train))
        .map(lambda x, y: preprocess(x, y))
        .batch(256)
        .prefetch(buffer_size=AUTOTUNE)
    )

    validation_dataset = (
        tf.data.Dataset.from_tensor_slices((paths_val, labels_val))
        .map(lambda x, y: preprocess(x, y))
        .batch(256)
        .prefetch(buffer_size=AUTOTUNE)
    )

    return train_dataset, validation_dataset


def get_test_dataset():
    """
    Create a test dataset from the Oxford-IIIT Pet Dataset.

    Returns:
        tf.data.Dataset: Test dataset.
    """
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "oxford-iiit-pet"
    df = pd.read_csv(data_dir / "annotations" / "test.txt", sep=" ")
    df.columns = ["path", "species", "breed", "ID"]
    df["breed"] = df.breed.apply(lambda x: x - 1)
    df["path"] = df["path"].apply(lambda x: str(data_dir / f"images/{x}.jpg"))
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((df["path"], df["breed"]))
        .map(lambda x, y: preprocess(x, y))
        .batch(256)
    )
    return test_dataset


def get_train_val_datasets():
    """
    Create preprocessed train and validation datasets from the Oxford-IIIT Pet Dataset.

    Returns:
        tuple: Preprocessed train and validation tf.data.Dataset objects.
    """
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "oxford-iiit-pet"
    df = pd.read_csv(data_dir / "annotations" / "trainval.txt", sep=" ")
    df.columns = ["path", "species", "breed", "ID"]
    df["breed"] = df.breed.apply(lambda x: x - 1)
    df["path"] = df["path"].apply(lambda x: str(data_dir / f"images/{x}.jpg"))
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        df["path"], df["breed"], test_size=0.2, random_state=0
    )
    train_dataset, val_dataset = get_datasets(paths_train, labels_train, paths_val, labels_val)
    train_dataset_preprocessed = train_dataset.map(lambda x, y: (x / 255.0, y))
    val_dataset_preprocessed = val_dataset.map(lambda x, y: (x / 255.0, y))
    return train_dataset_preprocessed, val_dataset_preprocessed
