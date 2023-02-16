from typing import List

import numpy as np
import tensorflow as tf
from keras import Model

from base.constants import NUM_ENSEMBLE_MEMBERS
from base.model import build_and_compile_model


def train_ensemble(train_images, train_labels):
    ensemble_model = []
    for ind in range(NUM_ENSEMBLE_MEMBERS):
        member = build_and_compile_model()
        print(f"Train model {ind:02}")
        member.fit(train_images, train_labels, epochs=10)
        ensemble_model.append(member)
    return ensemble_model


def get_ensemble_predictions(ensemble: List[Model], images):
    ensemble_predictions = tf.stack(
        [model.predict(images) for model in ensemble],
        axis=0,
    )
    return np.mean(ensemble_predictions, axis=0)
