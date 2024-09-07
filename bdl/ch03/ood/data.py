from pathlib import Path
from typing import Tuple

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE
DATA_ROOT = Path(__file__).parents[3] / "data" / "ch03" / "ood"
MODEL_DIR = Path(__file__).parents[3] / "models" / "ch03" / "ood"


def read_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / "oxford-iiit-pet/annotations/trainval.txt", sep=" ")
    df.columns = ["path", "species", "breed", "ID"]
    df["breed"] = df.breed - 1
    df["path"] = df["path"].apply(lambda x: str(DATA_ROOT / f"oxford-iiit-pet/images/{x}.jpg"))
    return df


def get_test_data() -> Tuple[tf.data.Dataset, pd.DataFrame]:
    df_test = pd.read_csv(DATA_ROOT / "oxford-iiit-pet/annotations/test.txt", sep=" ")
    df_test.columns = ["path", "species", "breed", "ID"]
    df_test["breed"] = df_test.breed - 1
    df_test["path"] = df_test["path"].apply(
        lambda x: str(DATA_ROOT / f"oxford-iiit-pet/images/{x}.jpg")
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((df_test["path"], df_test["breed"]))
        .map(lambda x, y: preprocess(x, y))
        .batch(256)
    )
    return test_dataset, df_test


def get_train_val_data() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    df = read_data()
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        df["path"], df["breed"], test_size=0.2, random_state=0
    )
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


@tf.function
def preprocess_image(filename):
    raw = tf.io.read_file(filename)
    image = tf.image.decode_png(raw, channels=3)
    return tf.image.resize(image, IMG_SIZE)


@tf.function
def preprocess(filename, label):
    return preprocess_image(filename), tf.one_hot(label, 2)
