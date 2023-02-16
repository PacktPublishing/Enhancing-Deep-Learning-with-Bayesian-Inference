import numpy as np

from base.constants import NUM_SUBSET, NUM_LEVELS, CORRUPTION_FUNCTIONS


def get_test_subset(test_images, test_labels):
    test_images_subset = test_images[:NUM_SUBSET]
    test_labels_subset = test_labels[:NUM_SUBSET]
    return test_images_subset, test_labels_subset


def get_corrupted_images(images):
    corrupted_images = []
    # loop over different corruption severities
    for corruption_severity in range(1, NUM_LEVELS + 1):
        corruption_type_batch = []
        # loop over different corruption types
        for corruption_type in CORRUPTION_FUNCTIONS:
            corrupted_image_batch = corruption_type(severity=corruption_severity, seed=0)(images=images)
            corruption_type_batch.append(corrupted_image_batch)
        corruption_type_batch = np.stack(corruption_type_batch, axis=0)
        corrupted_images.append(corruption_type_batch)
    corrupted_images = np.stack(corrupted_images, axis=0)

    return corrupted_images.reshape((-1, 32, 32, 3))
