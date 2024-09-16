from pathlib import Path
from typing import Optional, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


IMG_SIZE = (160, 160)
LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def preprocess_image(filename: tf.Tensor) -> tf.Tensor:
    raw = tf.io.read_file(filename)
    image = tf.image.decode_png(raw, channels=3)
    return tf.image.resize(image, IMG_SIZE)


@tf.function
def preprocess(filename: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return preprocess_image(filename), tf.one_hot(label, 2)

@click.command()
@click.option("--model-path", type=click.STRING, required=True)
@click.option("--output-path", type=click.STRING, required=False, default=None)
def main(model_path: str, output_path: Optional[str] = None):
    model = tf.keras.models.load_model(model_path)
    image, label = preprocess(str(Path(__file__).parent.parent.parent / "data" / "cat.png"), 0)
    label = tf.expand_dims(label, 0)
    epsilon = 0.5
    perturbation = get_adversarial_perturbation(image, label, model)
    image_perturbed = image + epsilon * perturbation
    cat_score = 1 - get_dog_score(image, model)
    dog_score = get_dog_score(image_perturbed, model)

    ax = plt.subplots(1, 3, figsize=(20, 10))[1]
    [ax.set_axis_off() for ax in ax.ravel()]
    ax[0].imshow(image.numpy().astype(int))
    ax[0].title.set_text("Original image")
    ax[0].text(
        0.5,
        -0.1,
        f'"Cat"\n {cat_score:.2%} confidence',
        size=12,
        ha="center",
        transform=ax[0].transAxes,
    )
    ax[1].imshow(perturbation)
    ax[1].title.set_text("Perturbation added to the image\n(multiplied by epsilon)")
    ax[2].imshow(image_perturbed.numpy().astype(int))
    ax[2].title.set_text("Perturbed image")
    ax[2].text(
        0.5,
        -0.1,
        f'"Dog"\n {dog_score:.2%} confidence',
        size=12,
        ha="center",
        transform=ax[2].transAxes,
    )
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def get_adversarial_perturbation(image, label, model):
    image = tf.expand_dims(image, 0)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = LOSS(label, prediction)

    gradient = tape.gradient(loss, image)
    return tf.sign(gradient)[0]


def get_dog_score(image: np.ndarray, model) -> float:
    scores = tf.nn.softmax(model.predict(np.expand_dims(image, 0)), axis=1).numpy()[0]
    return scores[1]


if __name__ == "__main__":
    main()
