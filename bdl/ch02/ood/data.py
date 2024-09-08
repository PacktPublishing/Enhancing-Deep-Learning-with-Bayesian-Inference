from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess_data(
    file_path: str, is_test: bool = False
) -> Union[Tuple[pd.Series, pd.Series, pd.Series, pd.Series], pd.DataFrame]:
    df = pd.read_csv(file_path, sep=" ")
    df.columns = ["path", "species", "breed", "ID"]
    df["breed"] = df.breed.apply(lambda x: x - 1)
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "oxford-iiit-pet" / "images"
    df["path"] = df["path"].apply(lambda x: str(data_dir / f"{x}.jpg"))
    if not is_test:
        return train_test_split(df["path"], df["breed"], test_size=0.2, random_state=0)
    return df


@tf.function
def preprocess_image(filename: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(filename)
    image = tf.image.decode_png(raw, channels=3)
    return tf.image.resize(image, IMG_SIZE)


@tf.function
def preprocess(filename: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return preprocess_image(filename), tf.one_hot(label, 2)


def create_dataset(
    paths: Union[pd.Series, tf.Tensor], labels: Union[pd.Series, tf.Tensor]
) -> tf.data.Dataset:
    return (
        tf.data.Dataset.from_tensor_slices((paths, labels))
        .map(lambda x, y: preprocess(x, y))
        .batch(256)
        .prefetch(buffer_size=AUTOTUNE)
    )
