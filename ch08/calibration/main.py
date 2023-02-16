import numpy as np
import tensorflow as tf

from base.accuracy import plot_accuracy
from base.bbb import build_and_compile_model_bbb, get_bbb_predictions
from base.calibration import plot_calibration
from base.constants import NUM_LEVELS, NUM_TYPES, NUM_SUBSET, NUM_INFERENCES_BBB, NUM_ENSEMBLE_MEMBERS
from base.corruption import get_corrupted_images, get_test_subset
from base.corruption_image import plot_corrupted_images
from base.ensemble import train_ensemble, get_ensemble_predictions
from base.model import build_and_compile_model
from base.scores import Scores


def main():
    cifar = tf.keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar.load_data()
    num_train_examples = train_images.shape[0]

    print("training vanilla")
    vanilla_model = build_and_compile_model()
    vanilla_model.fit(train_images, train_labels, epochs=10)
    vanilla_model.save("base_vanilla")
    # vanilla_model = tf.keras.models.load_model("base_vanilla")

    print("training ensemble")
    ensemble_model = train_ensemble(train_images, train_labels)
    for idx, m in enumerate(ensemble_model):
        m.save(f"base_ensemble_{idx}")
    # ensemble_model = [tf.keras.models.load_model(f"base_ensemble_{idx}") for idx in range(NUM_ENSEMBLE_MEMBERS)]

    print("training bbb")
    bbb_model = build_and_compile_model_bbb(num_train_examples)
    bbb_model.fit(train_images, train_labels, epochs=15)
    bbb_model.save("base_bbb")
    # bbb_model = tf.keras.models.load_model("base_bbb")

    print("Getting corrupted images")
    test_images_subset, test_labels_subset = get_test_subset(test_images, test_labels)
    # corrupted_images = get_corrupted_images(test_images_subset)
    # np.save("corrupted.npy", corrupted_images)
    corrupted_images = np.load("corrupted.npy")

    print("Getting vanilla scores")
    vanilla = get_vanilla_scores(corrupted_images, test_images_subset, vanilla_model)
    print("Getting ensemble scores")
    ensemble = get_ensemble_scores(corrupted_images, ensemble_model, test_images_subset)
    print("Getting bbb scores")
    bbb = get_bbb_scores(bbb_model, corrupted_images, test_images_subset)

    print("Creating plots")
    plot_corrupted_images(corrupted_images, vanilla, ensemble, bbb)
    plot_accuracy(test_labels_subset, vanilla, ensemble, bbb)
    plot_calibration(test_labels_subset, vanilla, ensemble, bbb)


def get_bbb_scores(bbb_model, corrupted_images, test_images_subset):
    # Get predictions on original images
    bbb_predictions = get_bbb_predictions(bbb_model, test_images_subset, NUM_INFERENCES_BBB)
    # Get predictions on corrupted images
    bbb_predictions_on_corrupted = get_bbb_predictions(bbb_model, corrupted_images, NUM_INFERENCES_BBB)
    bbb_predictions_on_corrupted = bbb_predictions_on_corrupted.reshape((NUM_LEVELS, NUM_TYPES, NUM_SUBSET, -1))
    bbb = Scores(bbb_predictions, bbb_predictions_on_corrupted)
    return bbb


def get_ensemble_scores(corrupted_images, ensemble_model, test_images_subset):
    # Get predictions on original images
    ensemble_predictions = get_ensemble_predictions(ensemble_model, test_images_subset)
    # Get predictions on corrupted images
    ensemble_predictions_on_corrupted = get_ensemble_predictions(ensemble_model, corrupted_images)
    ensemble_predictions_on_corrupted = ensemble_predictions_on_corrupted.reshape(
        (NUM_LEVELS, NUM_TYPES, NUM_SUBSET, -1)
    )
    ensemble = Scores(ensemble_predictions, ensemble_predictions_on_corrupted)
    return ensemble


def get_vanilla_scores(corrupted_images, test_images_subset, vanilla_model):
    # Get predictions on original images
    vanilla_predictions = vanilla_model.predict(test_images_subset)
    # Get predictions on corrupted images
    vanilla_predictions_on_corrupted = vanilla_model.predict(corrupted_images)
    vanilla_predictions_on_corrupted = vanilla_predictions_on_corrupted.reshape((NUM_LEVELS, NUM_TYPES, NUM_SUBSET, -1))
    vanilla = Scores(vanilla_predictions, vanilla_predictions_on_corrupted)
    return vanilla


def get_classes_and_scores(model_predictions):
    model_predicted_classes = np.argmax(model_predictions, axis=-1)
    model_scores = np.max(model_predictions, axis=-1)
    return model_predicted_classes, model_scores


if __name__ == "__main__":
    main()
