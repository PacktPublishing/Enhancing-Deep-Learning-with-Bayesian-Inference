"""
Adversarial Attack and Monte Carlo Dropout Evaluation

This script trains a neural network model for image classification, performs
adversarial attacks using the Fast Gradient Sign Method (FGSM), and evaluates
the model's performance using both standard predictions and Monte Carlo dropout.

The script includes functions for:
- Performing Monte Carlo dropout inference
- Calculating mean softmax values for predictions
- Training the model
- Generating adversarial examples
- Evaluating model performance on standard and adversarial inputs

The main function orchestrates the entire process, including model training,
adversarial example generation, and performance evaluation.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from cleverhans.tf2.attacks.fast_gradient_method import (
    fast_gradient_method as fgsm,
)
from keras.metrics import CategoricalAccuracy
from tqdm import tqdm

from ch08.adversarial.data import get_test_dataset, get_train_val_datasets
from ch08.adversarial.model import get_model


def mc_dropout(model, images, n_inference: int = 50):
    """
    Perform Monte Carlo dropout inference on the given model.

    Args:
        model: The neural network model.
        images: Input images to perform inference on.
        n_inference (int): Number of inference passes to perform.

    Returns:
        numpy.ndarray: Stacked predictions from multiple inference passes.
    """
    return np.swapaxes(np.stack([model(images, training=True) for _ in range(n_inference)]), 0, 1)


def get_mean_softmax_value(predictions) -> float:
    """
    Calculate the mean of the maximum softmax values for a set of predictions.

    Args:
        predictions: Model predictions.

    Returns:
        float: Mean of the maximum softmax values.
    """
    mean_softmax = tf.nn.softmax(predictions, axis=1)
    max_softmax = np.max(mean_softmax, axis=1)
    mean_max_softmax = max_softmax.mean()
    return mean_max_softmax


def get_mean_softmax_value_mc(predictions) -> float:
    """
    Calculate the mean of the maximum softmax values for Monte Carlo predictions.

    Args:
        predictions: Monte Carlo predictions.

    Returns:
        float: Mean of the maximum softmax values.
    """
    predictions_np = np.stack(predictions)
    predictions_np_mean = predictions_np.mean(axis=1)
    return get_mean_softmax_value(predictions_np_mean)


def main():
    """
    Main function to train the model, perform adversarial attacks, and evaluate results.
    """
    train_dataset_preprocessed, val_dataset_preprocessed = get_train_val_datasets()

    model_save_path = Path(__file__).parent / "saved_models" / "pet_classifier"
    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset_preprocessed,
        epochs=200,
        validation_data=val_dataset_preprocessed,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_save_path / "best_model.keras"),
                save_best_only=True,
                monitor="val_accuracy",
                mode="max",
                verbose=1,
            )
        ],
    )

    # Save the model
    model.save(model_save_path / "model.keras")
    print(f"Model saved to {model_save_path}")

    predictions_standard, predictions_fgsm, labels = [], [], []
    test_dataset = get_test_dataset()
    for imgs, labels_batch in test_dataset:
        imgs /= 255.0
        predictions_standard.extend(model.predict(imgs))
        imgs_adv = fgsm(model, imgs, 0.01, np.inf)
        predictions_fgsm.extend(model.predict(imgs_adv))
        labels.extend(labels_batch)

    accuracy_standard = CategoricalAccuracy()(labels, predictions_standard).numpy()
    accuracy_fgsm = CategoricalAccuracy()(labels, predictions_fgsm).numpy()
    print(f"{accuracy_standard=:.2%}, {accuracy_fgsm=:.2%}")

    predictions_standard_mc, predictions_fgsm_mc, labels = [], [], []
    for imgs, labels_batch in tqdm(test_dataset):
        imgs /= 255.0
        predictions_standard_mc.extend(mc_dropout(model, imgs, 50))
        imgs_adv = fgsm(model, imgs, 0.01, np.inf)
        predictions_fgsm_mc.extend(mc_dropout(model, imgs_adv, 50))
        labels.extend(labels_batch)

    accuracy_standard_mc = CategoricalAccuracy()(
        labels, np.stack(predictions_standard_mc).mean(axis=1)
    ).numpy()
    accuracy_fgsm_mc = CategoricalAccuracy()(
        labels, np.stack(predictions_fgsm_mc).mean(axis=1)
    ).numpy()
    print(f"{accuracy_standard_mc=:.2%}, {accuracy_fgsm_mc=:.2%}")
    # accuracy_standard_mc=86.60%, accuracy_fgsm_mc=80.75%

    mean_standard = get_mean_softmax_value(predictions_standard)
    mean_fgsm = get_mean_softmax_value(predictions_fgsm)
    mean_standard_mc = get_mean_softmax_value_mc(predictions_standard_mc)
    mean_fgsm_mc = get_mean_softmax_value_mc(predictions_fgsm_mc)
    print(f"{mean_standard=:.2%}, {mean_fgsm=:.2%}")
    print(f"{mean_standard_mc=:.2%}, {mean_fgsm_mc=:.2%}")
    # mean_standard=89.58%, mean_fgsm=89.91%
    # mean_standard_mc=89.48%, mean_fgsm_mc=85.25%


if __name__ == "__main__":
    main()
