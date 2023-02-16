import cv2
import numpy as np
from matplotlib import pyplot as plt

from base.constants import CLASS_NAMES, NUM_LEVELS, NUM_TYPES, NUM_SUBSET
from base.scores import Scores


def plot_corrupted_images(corrupted_images: np.ndarray, vanilla: Scores, ensemble: Scores, bbb: Scores):
    plot_images = corrupted_images.reshape((NUM_LEVELS, NUM_TYPES, NUM_SUBSET, 32, 32, 3))
    # Index of the selected images
    ind_image = 9
    # Define figure
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(16, 10))
    # Loop over corruption levels
    for ind_level in range(NUM_LEVELS):
        # Loop over corruption types
        for ind_type in range(3):
            # Plot slightly upscaled image for easier inspection
            image = plot_images[ind_level, ind_type, ind_image, ...]
            image_upscaled = cv2.resize(image, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
            axes[ind_type, ind_level].imshow(image_upscaled)
            # Get score and class predicted by vanilla model
            vanilla_score = vanilla.max_scores_on_corrupted[ind_level, ind_type, ind_image, ...]
            vanilla_prediction = vanilla.predicted_classes_on_corrupted[ind_level, ind_type, ind_image, ...]
            # Get score and class predicted by ensemble model
            ensemble_score = ensemble.max_scores_on_corrupted[ind_level, ind_type, ind_image, ...]
            ensemble_prediction = ensemble.predicted_classes_on_corrupted[ind_level, ind_type, ind_image, ...]
            # Get score and class predicted by BBB model
            bbb_score = bbb.max_scores_on_corrupted[ind_level, ind_type, ind_image, ...]
            bbb_prediction = bbb.predicted_classes_on_corrupted[ind_level, ind_type, ind_image, ...]
            # Plot prediction info in title
            title_text = (
                f"Vanilla: {vanilla_score:.3f} "
                + f"[{CLASS_NAMES[vanilla_prediction]}] \n"
                + f"Ensemble: {ensemble_score:.3f} "
                + f"[{CLASS_NAMES[ensemble_prediction]}] \n"
                + f"BBB: {bbb_score:.3f} "
                + f"[{CLASS_NAMES[bbb_prediction]}]"
            )
            axes[ind_type, ind_level].set_title(title_text, fontsize=14)
            # Remove axes ticks and labels
            axes[ind_type, ind_level].axis("off")
    fig.tight_layout()
    plt.show()
