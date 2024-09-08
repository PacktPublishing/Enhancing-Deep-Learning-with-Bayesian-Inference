from typing import Callable

import ddu_dirty_mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_keras
from sklearn.metrics import roc_auc_score

from bdl.ch07.uncertainty_image_classification.data import get_data
from bdl.ch07.uncertainty_image_classification.model import get_model
from bdl.ch07.uncertainty_image_classification.uncertainty import (
    data_uncertainty,
    knowledge_uncertainty,
    total_uncertainty,
)


def auc_id_and_amb_vs_ood(uncertainty: Callable, preds_id, preds_ood, preds_amb):
    scores_id = uncertainty(preds_id)
    scores_ood = uncertainty(preds_ood)
    scores_amb = uncertainty(preds_amb)
    scores_id = np.concatenate([scores_id, scores_amb])
    labels = np.concatenate([np.zeros_like(scores_id), np.ones_like(scores_ood)])
    return roc_auc_score(labels, np.concatenate([scores_id, scores_ood]))


def auc_id_vs_amb(uncertainty, preds_id, preds_amb):
    scores_id, scores_amb = uncertainty(preds_id), uncertainty(preds_amb)
    labels = np.concatenate([np.zeros_like(scores_id), np.ones_like(scores_amb)])
    return roc_auc_score(labels, np.concatenate([scores_id, scores_amb]))


def plot(preds_id, preds_ood, preds_amb):
    labels = ["In-distribution", "Out-of-distribution", "Ambiguous"]
    uncertainty_functions = [total_uncertainty, data_uncertainty, knowledge_uncertainty]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, uncertainty in zip(axes, uncertainty_functions):
        for scores, label in zip([preds_id, preds_ood, preds_amb], labels):
            ax.hist(uncertainty(scores), bins=20, label=label, alpha=0.8)
        ax.title.set_text(uncertainty.__name__.replace("_", " ").capitalize())
        ax.legend(loc="upper right")
    plt.legend()
    plt.savefig("uncertainty_types.png", dpi=300)
    plt.show()


def get_preds(model, test_imgs, ood_imgs, amb_imgs):
    preds_id = []
    preds_ood = []
    preds_amb = []
    for _ in range(50):
        preds_id.append(model(test_imgs))
        preds_ood.append(model(ood_imgs))
        preds_amb.append(model(amb_imgs))
    # format data such that we have it in shape n_images, n_predictions, n_classes
    preds_id = np.moveaxis(np.stack(preds_id), 0, 1)
    preds_ood = np.moveaxis(np.stack(preds_ood), 0, 1)
    preds_amb = np.moveaxis(np.stack(preds_amb), 0, 1)
    return preds_id, preds_ood, preds_amb


def main():
    train_imgs, train_labels, test_imgs, test_labels = get_data()
    model = get_model()

    model.compile(
        tf_keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        experimental_run_tf_function=False,
    )
    model.fit(x=train_imgs, y=train_labels, validation_data=(test_imgs, test_labels), epochs=50)

    (_, _), (ood_imgs, _) = tf.keras.datasets.fashion_mnist.load_data()
    ood_imgs = np.expand_dims(ood_imgs / 255.0, -1)

    ambiguous_mnist_test = ddu_dirty_mnist.AmbiguousMNIST(
        ".", train=False, download=True, normalize=False, noise_stddev=0
    )
    amb_imgs = ambiguous_mnist_test.data.numpy().reshape(60000, 28, 28, 1)[:10000]

    preds_id, preds_ood, preds_amb = get_preds(model, test_imgs, ood_imgs, amb_imgs)
    plot(preds_id, preds_ood, preds_amb)

    print(f"{auc_id_and_amb_vs_ood(total_uncertainty)=:.2%}")
    print(f"{auc_id_and_amb_vs_ood(knowledge_uncertainty)=:.2%}")
    print(f"{auc_id_and_amb_vs_ood(data_uncertainty)=:.2%}")
    # output:
    # auc_id_and_amb_vs_ood(total_uncertainty)=91.81%
    # auc_id_and_amb_vs_ood(knowledge_uncertainty)=98.87%
    # auc_id_and_amb_vs_ood(data_uncertainty)=84.29%

    print(f"{auc_id_vs_amb(total_uncertainty)=:.2%}")
    print(f"{auc_id_vs_amb(knowledge_uncertainty)=:.2%}")
    print(f"{auc_id_vs_amb(data_uncertainty)=:.2%}")
    # output:
    # auc_id_vs_amb(total_uncertainty)=94.71%
    # auc_id_vs_amb(knowledge_uncertainty)=87.06%
    # auc_id_vs_amb(data_uncertainty)=95.21%


if __name__ == "__main__":
    main()
