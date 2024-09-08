"""
This module contains functions for training and evaluating a ResNet50-based model
for pet breed classification using the Oxford-IIIT Pet Dataset.

The main components include:
- Model definition using a pre-trained ResNet50 as the base
- Model training and evaluation
- Accuracy calculation

The main() function orchestrates the entire process, from data loading to final
accuracy reporting.
"""

from pathlib import Path

import pandas as pd
import tensorflow as tf

from ch03.ood.data import IMG_SIZE, create_dataset, load_and_preprocess_data


def get_model() -> tf.keras.Model:
    """
    Create and return a ResNet50-based model for pet breed classification.

    Returns
    -------
    tf.keras.Model
        A Keras model with ResNet50 as the base and additional layers for classification.
    """
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.ResNet50(
        input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(2)(x)
    return tf.keras.Model(inputs, outputs)


def evaluate_model(model: tf.keras.Model, test_dataset: tf.data.Dataset) -> tf.Tensor:
    """
    Evaluate the model on the test dataset and return predicted labels.

    Parameters
    ----------
    model : tf.keras.Model
        The trained Keras model to evaluate.
    test_dataset : tf.data.Dataset
        The test dataset to evaluate the model on.

    Returns
    -------
    tf.Tensor
        Predicted labels for the test dataset.
    """
    test_predictions = model.predict(test_dataset)
    softmax_scores = tf.nn.softmax(test_predictions, axis=1)
    return tf.argmax(softmax_scores, axis=1)


def calculate_accuracy(df_test: pd.DataFrame, predicted_labels: tf.Tensor) -> float:
    """
    Calculate the accuracy of the model predictions.

    Parameters
    ----------
    df_test : pd.DataFrame
        DataFrame containing the true labels for the test dataset.
    predicted_labels : tf.Tensor
        Predicted labels from the model.

    Returns
    -------
    float
        The accuracy of the model predictions.
    """
    df_test["predicted_label"] = predicted_labels
    df_test["prediction_correct"] = df_test.apply(lambda x: x.predicted_label == x.breed, axis=1)
    return df_test.prediction_correct.value_counts(True)[True]


def main() -> None:
    """
    Main function to orchestrate the binary cat vs dog model training and evaluation process.

    This function loads the data, creates and trains the model, evaluates it on the test set,
    and prints the final accuracy.
    """
    annotation_dir = (
        Path(__file__).parent.parent.parent.parent / "data" / "oxford-iiit-pet" / "annotations"
    )
    paths_train, paths_val, labels_train, labels_val = load_and_preprocess_data(
        annotation_dir / "trainval.txt"
    )

    train_dataset = create_dataset(paths_train, labels_train)
    validation_dataset = create_dataset(paths_val, labels_val)

    model = get_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(train_dataset, epochs=3, validation_data=validation_dataset)

    model.save("model.keras")

    df_test = load_and_preprocess_data(annotation_dir / "test.txt", is_test=True)
    test_dataset = create_dataset(df_test["path"], df_test["breed"])

    predicted_labels = evaluate_model(model, test_dataset)

    accuracy = calculate_accuracy(df_test, predicted_labels)
    print(f"Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
