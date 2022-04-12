import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn import datasets
from sklearn.model_selection import train_test_split

from bdl.pbp.pbp import PBP
from bdl.pbp.utils import normalize, get_mean_std_x_y, ensure_input, build_layers

NUM_EPOCHS = 40
RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def get_data():
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    return X_train, y_train, X_test, y_test


def fit(X_train, y_train, n_epochs: int = 1):
    units = [50, 50, 1]
    layers = build_layers(X_train.shape[1], units, tf.float32)
    pbp = PBP(layers)
    x, y = normalize(X_train, y_train, units[-1])
    pbp.fit(x, y, batch_size=1, n_epochs=n_epochs)
    return pbp


def predict(pbp, X_test, y_test, X_train, y_train):
    # normalise test set
    mean_X_train, mean_y_train, std_X_train, std_y_train = get_mean_std_x_y(X_train, y_train)
    # We normalize the test set
    X_test = (X_test - np.full(X_test.shape, mean_X_train)) / np.full(X_test.shape, std_X_train)

    # perform inference on normalised data
    X_test = ensure_input(X_test, tf.float32, X_test.shape[1])
    m, v = pbp.predict(X_test)
    v_noise = (pbp.beta / (pbp.alpha - 1) * std_y_train**2)
    # add mean and std back to m and v
    m = m * std_y_train + mean_y_train
    v = v * std_y_train**2

    # transform back to original space
    m = np.squeeze(m.numpy())
    v = np.squeeze(v.numpy())
    v_noise = np.squeeze(v_noise.numpy().reshape(-1, 1))

    # calculate rmse
    rmse = np.sqrt(np.mean((y_test - m) ** 2))
    print(f"{rmse=}")
    # calculate log-likelihood
    test_ll = np.mean(
        -0.5 * np.log(2 * math.pi * v)
        - 0.5 * (y_test - m) ** 2 / v
    )
    print(f"{test_ll=}")
    # calculate log-likelihood with v_noise
    test_ll_with_vnoise = np.mean(
        -0.5 * np.log(2 * math.pi * (v + v_noise))
        - 0.5 * (y_test - m) ** 2 / (v + v_noise)
    )
    print(f"{test_ll_with_vnoise=}")
    return m, v, rmse, test_ll, test_ll_with_vnoise


def plot(X_test, y_test, m, v, num_epochs, rmse):
    id = np.arange(X_test.shape[0])
    plt.figure(figsize=(15, 15))
    plt.plot(
        id, y_test, linestyle="", marker=".", label="data"
    )
    plt.plot(id, m, alpha=0.5, label="predict mean")
    plt.fill_between(
        id,
        m + np.sqrt(v),
        m - np.sqrt(v),
        alpha=0.5,
        label="credible interval",
        )
    plt.xlabel("data id")
    plt.ylabel("target: home price")
    plt.ylim([-10, 60])
    plt.title(f"Tensorflow, {num_epochs} epochs, random seed: {RANDOM_SEED}, rmse: {rmse}")
    plt.legend()
