from pathlib import Path
from typing import Optional

import click
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from bdl.ch03.ood.data import preprocess, MODEL_DIR

LOSS = tf.keras.losses.BinaryCrossentropy(from_logits=True)
DATA_ROOT = Path(__file__).parents[3] / "data" / "ch03" / "adversarial"


@click.command()
@click.option("--output-path", type=click.STRING, required=False, default=None)
def main(output_path: Optional[str] = None):
    model = tf.keras.models.load_model(MODEL_DIR)
    image, label = preprocess(str(DATA_ROOT / "cat.png"), 0)
    label = tf.expand_dims(label, 0)
    epsilon = 0.1
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
        -.1,
        f"\"Cat\"\n {cat_score:.2%} confidence",
        size=12,
        ha="center",
        transform=ax[0].transAxes
    )
    ax[1].imshow(perturbation)
    ax[1].title.set_text(
        "Perturbation added to the image\n(multiplied by epsilon)"
    )
    ax[2].imshow(image_perturbed.numpy().astype(int))
    ax[2].title.set_text("Perturbed image")
    ax[2].text(
        0.5,
        -.1,
        f"\"Dog\"\n {dog_score:.2%} confidence",
        size=12,
        ha="center",
        transform=ax[2].transAxes
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
  scores = tf.nn.softmax(
      model.predict(np.expand_dims(image, 0)), axis=1
  ).numpy()[0]
  return scores[1]


if __name__ == '__main__':
    main()
