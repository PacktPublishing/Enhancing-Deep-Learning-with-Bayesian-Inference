from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import click
from tensorflow.keras import datasets, layers, models


def get_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10))
    return model


def remove_signal(img: np.ndarray, num_lines: int) -> np.ndarray:
    img = img.copy()
    img[:num_lines] = 0
    return img


def get_dropout_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10))
    return model


def plot_predictions(predictions, imgs, output_filepath: Path):
    plt.figure(figsize=(10, 10))
    bbox_dict = dict(fill=True, facecolor="white", alpha=0.5, edgecolor="white", linewidth=0)
    for i in range(len(imgs)):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i], cmap="gray")
        prediction = predictions[i].max()
        label = np.argmax(predictions[i])
        plt.xlabel(f"{label} - {prediction:.2%}")
        plt.text(0, 3, f" {i+1}", bbox=bbox_dict)
    plt.savefig(output_filepath)


def fit_model(model, train_images, train_labels, test_images, test_labels):
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        train_images,
        train_labels,
        epochs=5,
        validation_data=(test_images, test_labels),
    )
    return model

@click.command()
@click.option("--output-dir", type=click.STRING, required=True)
def main(output_dir: str):
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # Train regular model
    model = get_model()
    model = fit_model(model, train_images, train_labels, test_images, test_labels)

    # Generate perturbed images
    img = test_images[0]  # Assuming we're using the first test image
    imgs = []
    for i in range(28):
        img_perturbed = remove_signal(img, i)
        if np.array_equal(img, img_perturbed):
            continue
        imgs.append(img_perturbed)
        if img_perturbed.sum() == 0:
            break

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make predictions and plot results
    softmax_predictions = tf.nn.softmax(model(np.expand_dims(imgs, -1)), axis=1).numpy()
    plot_predictions(softmax_predictions, imgs, output_dir / "softmax.png")

    # Train dropout model
    dropout_model = get_dropout_model()
    dropout_model = fit_model(dropout_model, train_images, train_labels, test_images, test_labels)

    # Make predictions with dropout model
    imgs_np = np.expand_dims(imgs, -1)
    predictions = np.array(
        [tf.nn.softmax(dropout_model(imgs_np, training=True), axis=1) for _ in range(100)]
    )
    predictions_mean = np.mean(predictions, axis=0)
    plot_predictions(predictions_mean, imgs, output_dir / "mc_dropout.png")


if __name__ == "__main__":
    main()
