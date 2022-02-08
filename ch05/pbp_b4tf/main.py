import math
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from ch05.pbp_b4tf.pbp import PBP

np.random.seed(0)
tf.random.set_seed(0)


def main():
    X_test, X_train, y_train, y_test, x_scaler, y_scaler = get_data()

    pbp = fit(X_train, y_train)

    m, v, y_test = predict(X_test, pbp, x_scaler, y_scaler, y_test)

    plot(X_test, m, v, y_scaler, y_test)


def fit(X_train, y_train):
    print("Fitting..")
    pbp = PBP([50, 50, 1], input_shape=X_train.shape[1])
    pbp.fit(X_train, y_train, batch_size=8)
    return pbp


def plot(X_test, m, v, y_scaler, y_test):
    print("Plotting..")
    id = np.arange(X_test.shape[0])
    plt.figure(figsize=(15, 15))
    plt.plot(
        id, y_scaler.inverse_transform(y_test), linestyle="", marker=".", label="data"
    )
    plt.plot(id, y_scaler.inverse_transform(m), alpha=0.5, label="predict mean")
    plt.fill_between(
        id,
        y_scaler.inverse_transform(m + tf.sqrt(v)).squeeze(),
        y_scaler.inverse_transform(m - tf.sqrt(v)).squeeze(),
        alpha=0.5,
        label="credible interval",
    )
    plt.xlabel("data id")
    plt.ylabel("target")
    plt.legend()
    plt.savefig(Path(__file__).parent / "pbp_results.png")


def predict(pbp, X_test, y_test, x_scaler, y_scaler, normalize: bool = True):
    if normalize:
        X_test = x_scaler.fit_transform(X_test)
        y_test = y_scaler.fit_transform(y_test.reshape(-1, 1))
    m, v = pbp.predict(X_test)
    m_squeezed, v_squeezed = tf.squeeze(m), tf.squeeze(v)
    rmse = np.sqrt(np.mean((y_test - m_squeezed) ** 2))
    test_ll = np.mean(
        -0.5 * np.log(2 * math.pi * v_squeezed)
        - 0.5 * (y_test - m_squeezed) ** 2 / v_squeezed
    )
    print(f"{rmse=}, {test_ll=}")
    return m, v


def get_data():
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    return X_test, X_train, y_train, y_test, x_scaler, y_scaler


if __name__ == "__main__":
    main()
