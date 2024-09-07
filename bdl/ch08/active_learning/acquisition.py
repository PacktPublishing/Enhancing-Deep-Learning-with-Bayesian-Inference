from typing import Callable

import numpy as np
from tensorflow.keras.models import Model
from tqdm import tqdm

from bdl.active_learning.metrics import knowledge_uncertainty


def get_mc_predictions(model: Model, n_iter: int, x_train: np.ndarray) -> np.ndarray:
    preds = []
    for _ in tqdm(range(n_iter)):
        preds_iter = [model(batch, training=True) for batch in np.array_split(x_train, 6)]
        preds.append(np.concatenate(preds_iter))
    # format data such that we have n_images, n_predictions, n_classes
    preds = np.moveaxis(np.stack(preds), 0, 1)
    return preds


def acquire_knowledge_uncertainty(
    x_train: np.ndarray, n_samples: int, model: Model, n_iter: int, *args, **kwargs
):
    preds = get_mc_predictions(model, n_iter, x_train)
    ku = knowledge_uncertainty(preds)
    return np.argsort(ku, axis=-1)[-n_samples:]


def acquire_random(x_train: np.ndarray, n_samples: int, *args, **kwargs):
    return np.random.randint(low=0, high=len(x_train), size=n_samples)


def acquisition_factory(acquisition_type: str) -> Callable:
    if acquisition_type == "knowledge_uncertainty":
        return acquire_knowledge_uncertainty
    if acquisition_type == "random":
        return acquire_random
    raise ValueError(f"Acquisition type {acquisition_type} not supported")
