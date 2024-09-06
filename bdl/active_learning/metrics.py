import numpy as np
import tensorflow as tf



def total_uncertainty(
    preds: np.ndarray, epsilon: float = 1e-10
) -> np.ndarray:
    mean_preds = np.mean(preds, axis=1)
    log_preds = -np.log(mean_preds + epsilon)
    return np.sum(mean_preds * log_preds, axis=1)


def data_uncertainty(preds: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    log_preds = -np.log(preds + epsilon)
    return np.mean(np.sum(preds * log_preds, axis=2), axis=1)


def knowledge_uncertainty(
    preds: np.ndarray, epsilon: float = 1e-10
) -> np.ndarray:
    return total_uncertainty(preds, epsilon) - data_uncertainty(preds, epsilon)


def get_accuracy(y_test: np.ndarray, preds: np.ndarray) -> float:
    acc = tf.keras.metrics.CategoricalAccuracy()
    acc.update_state(preds, y_test)
    return acc.result().numpy() * 100