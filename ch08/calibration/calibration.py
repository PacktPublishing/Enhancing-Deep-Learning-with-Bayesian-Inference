import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from ch08.calibration.constants import NUM_BINS, NUM_TYPES, NUM_LEVELS
from ch08.calibration.scores import Scores


def expected_calibration_error(
    pred_correct,
    pred_score,
    n_bins=5,
):
    """Compute expected calibration error.
    ----------
    pred_correct : array-like of shape (n_samples,)
        Whether the prediction is correct or not
    pred_score : array-like of shape (n_samples,)
        Confidence in the prediction
    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval.
    """
    # Convert from bool to integer (makes counting easier)
    pred_correct = pred_correct.astype(np.int32)

    # Create bins and assign prediction scores to bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], pred_score)

    # Count number of samples and correct predictions per bin
    bin_true_counts = np.bincount(binids, weights=pred_correct, minlength=len(bins))
    bin_counts = np.bincount(binids, minlength=len(bins))

    # Calculate sum of confidence scores per bin
    bin_probs = np.bincount(binids, weights=pred_score, minlength=len(bins))

    # Identify bins that contain samples
    nonzero = bin_counts != 0
    # Calculate accuracy for every bin
    bin_acc = bin_true_counts[nonzero] / bin_counts[nonzero]
    # Calculate average confidence scores per bin
    bin_conf = bin_probs[nonzero] / bin_counts[nonzero]

    return np.average(np.abs(bin_acc - bin_conf), weights=bin_counts[nonzero])


def plot_calibration(test_labels_subset, vanilla: Scores, ensemble: Scores, bbb: Scores):
    vanilla_cal = expected_calibration_error(
        test_labels_subset.flatten() == vanilla.predicted_classes,
        vanilla.max_scores,
        n_bins=NUM_BINS,
    )
    ensemble_cal = expected_calibration_error(
        test_labels_subset.flatten() == ensemble.predicted_classes,
        ensemble.max_scores,
        n_bins=NUM_BINS,
    )
    bbb_cal = expected_calibration_error(
        test_labels_subset.flatten() == bbb.predicted_classes,
        bbb.max_scores,
        n_bins=NUM_BINS,
    )
    calibration = [
        {"model_name": "vanilla", "type": 0, "level": 0, "calibration": vanilla_cal},
        {
            "model_name": "ensemble",
            "type": 0,
            "level": 0,
            "calibration": ensemble_cal,
        },
        {"model_name": "bbb", "type": 0, "level": 0, "calibration": bbb_cal},
    ]

    for ind_type in range(NUM_TYPES):
        for ind_level in range(NUM_LEVELS):
            # Calculate calibration error for vanilla model
            vanilla_cal_on_corrupted = expected_calibration_error(
                test_labels_subset.flatten() == vanilla.predicted_classes_on_corrupted[ind_level, ind_type, :],
                vanilla.max_scores_on_corrupted[ind_level, ind_type, :],
            )
            calibration.append(
                {
                    "model_name": "vanilla",
                    "type": ind_type + 1,
                    "level": ind_level + 1,
                    "calibration": vanilla_cal_on_corrupted,
                }
            )

            # Calculate calibration error for ensemble model
            ensemble_cal_on_corrupted = expected_calibration_error(
                test_labels_subset.flatten() == ensemble.predicted_classes_on_corrupted[ind_level, ind_type, :],
                ensemble.max_scores_on_corrupted[ind_level, ind_type, :],
            )
            calibration.append(
                {
                    "model_name": "ensemble",
                    "type": ind_type + 1,
                    "level": ind_level + 1,
                    "calibration": ensemble_cal_on_corrupted,
                }
            )

            # Calculate calibration error for BBB model
            bbb_cal_on_corrupted = expected_calibration_error(
                test_labels_subset.flatten() == bbb.predicted_classes_on_corrupted[ind_level, ind_type, :],
                bbb.max_scores_on_corrupted[ind_level, ind_type, :],
            )
            calibration.append(
                {
                    "model_name": "bbb",
                    "type": ind_type + 1,
                    "level": ind_level + 1,
                    "calibration": bbb_cal_on_corrupted,
                }
            )
    df = pd.DataFrame(calibration)
    plt.figure(dpi=100)
    sns.boxplot(data=df, x="level", y="calibration", hue="model_name")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("calibration.png")
    plt.show()
