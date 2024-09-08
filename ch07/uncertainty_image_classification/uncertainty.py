import numpy as np
from scipy.stats import entropy


def total_uncertainty(preds: np.ndarray) -> np.ndarray:
    return entropy(np.mean(preds, axis=1), axis=-1)


def data_uncertainty(preds: np.ndarray) -> np.ndarray:
    return np.mean(entropy(preds, axis=2), axis=-1)


def knowledge_uncertainty(preds: np.ndarray) -> np.ndarray:
    return total_uncertainty(preds) - data_uncertainty(preds)
