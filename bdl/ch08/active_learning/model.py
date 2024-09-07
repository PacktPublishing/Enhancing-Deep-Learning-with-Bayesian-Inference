from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Input, layers


def build_model():
    model = tf.keras.models.Sequential(
        [
            Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(4, 4), activation="relu"),
            layers.Conv2D(32, kernel_size=(4, 4), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callback(model_dir: Path):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        str(model_dir / "model.keras"),
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
    )
    return model_checkpoint_callback
