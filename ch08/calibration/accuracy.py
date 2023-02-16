import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from ch08.calibration.constants import NUM_TYPES, NUM_LEVELS
from ch08.calibration.scores import Scores


def plot_accuracy(test_labels_subset, vanilla: Scores, ensemble: Scores, bbb: Scores):
    vanilla_acc = accuracy_score(test_labels_subset.flatten(), vanilla.predicted_classes)
    ensemble_acc = accuracy_score(test_labels_subset.flatten(), ensemble.predicted_classes)
    bbb_acc = accuracy_score(test_labels_subset.flatten(), bbb.predicted_classes)
    accuracies = [
        {"model_name": "vanilla", "type": 0, "level": 0, "accuracy": vanilla_acc},
        {"model_name": "ensemble", "type": 0, "level": 0, "accuracy": ensemble_acc},
        {"model_name": "bbb", "type": 0, "level": 0, "accuracy": bbb_acc},
    ]
    for ind_type in range(NUM_TYPES):
        for ind_level in range(NUM_LEVELS):
            # Calculate accuracy for vanilla model
            vanilla_acc_on_corrupted = accuracy_score(
                test_labels_subset.flatten(),
                vanilla.predicted_classes_on_corrupted[ind_level, ind_type, :],
            )
            accuracies.append(
                {
                    "model_name": "vanilla",
                    "type": ind_type + 1,
                    "level": ind_level + 1,
                    "accuracy": vanilla_acc_on_corrupted,
                }
            )

            # Calculate accuracy for ensemble model
            ensemble_acc_on_corrupted = accuracy_score(
                test_labels_subset.flatten(),
                ensemble.predicted_classes_on_corrupted[ind_level, ind_type, :],
            )
            accuracies.append(
                {
                    "model_name": "ensemble",
                    "type": ind_type + 1,
                    "level": ind_level + 1,
                    "accuracy": ensemble_acc_on_corrupted,
                }
            )

            # Calculate accuracy for BBB model
            bbb_acc_on_corrupted = accuracy_score(
                test_labels_subset.flatten(),
                bbb.predicted_classes_on_corrupted[ind_level, ind_type, :],
            )
            accuracies.append(
                {
                    "model_name": "bbb",
                    "type": ind_type + 1,
                    "level": ind_level + 1,
                    "accuracy": bbb_acc_on_corrupted,
                }
            )
    df = pd.DataFrame(accuracies)
    plt.figure(dpi=100)
    sns.boxplot(data=df, x="level", y="accuracy", hue="model_name")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("accuracy.png")
    plt.show()
