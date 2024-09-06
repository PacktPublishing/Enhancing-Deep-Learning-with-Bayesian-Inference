from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def preprocess_image(filename):
  raw = tf.io.read_file(filename)
  image = tf.image.decode_png(raw, channels=3)
  return tf.image.resize(image, IMG_SIZE)


@tf.function
def preprocess(filename, label):
  return preprocess_image(filename), tf.one_hot(label, 2)
  

def get_datasets(paths_train, labels_train, paths_val, labels_val):
    train_dataset = (tf.data.Dataset.from_tensor_slices(
        (paths_train, labels_train)
    ).map(lambda x, y: preprocess(x, y))
    .batch(256)
    .prefetch(buffer_size=AUTOTUNE)
    )

    validation_dataset = (tf.data.Dataset.from_tensor_slices(
        (paths_val, labels_val))
    .map(lambda x, y: preprocess(x, y))
    .batch(256)
    .prefetch(buffer_size=AUTOTUNE)
    )

    return train_dataset, validation_dataset


def get_test_dataset():
    df_test = pd.read_csv("oxford-iiit-pet/annotations/test.txt", sep=" ")
    df_test.columns = ["path", "species", "breed", "ID"]
    df_test["breed"] = df_test.breed.apply(lambda x: x - 1)
    df_test["path"] = df_test["path"].apply(
        lambda x: str(Path(__file__).cwd() / "oxford-iiit-pet/images/{x}.jpg")
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (df_test["path"], df_test["breed"])
    ).map(lambda x, y: preprocess(x, y)).batch(256)
    return test_dataset


def get_train_val_datasets():
    df = pd.read_csv("oxford-iiit-pet/annotations/trainval.txt", sep=" ")
    df.columns = ["path", "species", "breed", "ID"]
    df["breed"] = df.breed.apply(lambda x: x - 1)
    df["path"] = df["path"].apply(
        lambda x: str(Path(__file__).cwd() / f"oxford-iiit-pet/images/{x}.jpg")
    )
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        df["path"], df["breed"], test_size=0.2, random_state=0
    )
    train_dataset, val_dataset = get_datasets(
        paths_train, labels_train, paths_val, labels_val
    )
    train_dataset_preprocessed = train_dataset.map(lambda x, y: (x / 255., y))
    val_dataset_preprocessed = val_dataset.map(lambda x, y: (x / 255., y))
    return train_dataset_preprocessed, val_dataset_preprocessed
