import math
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pbp import PBP


def get_data():
    # We load the boston housing dataset
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    # We obtain the features and the targets
    X = data[:, range(data.shape[1] - 1)]
    y = data[:, data.shape[1] - 1]
    # We create the train and test sets with 90% and 10% of the data
    permutation = np.random.choice(range(X.shape[0]), X.shape[0], replace=False)
    size_train = int(np.round(X.shape[0] * 0.9))
    index_train = permutation[0:size_train]
    index_test = permutation[size_train:]

    X_train = X[index_train, :]
    y_train = y[index_train]
    X_test = X[index_test, :]
    y_test = y[index_test]
    return X_train, y_train, X_test, y_test


def main():
    X, y = datasets.load_boston(
        return_X_y=True,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    scale_x = StandardScaler()
    scale_y = StandardScaler()
    X_train = scale_x.fit_transform(X_train)
    y_train = scale_y.fit_transform(y_train.reshape(-1, 1))
    print("Fitting..")
    pbp = PBP([50, 50, 1], input_shape=X_train.shape[1])
    pbp.fit(X_train, y_train, batch_size=8)

    print("Testing..")

    X_test = scale_x.fit_transform(X_test)
    y_test = scale_y.fit_transform(y_test.reshape(-1, 1))
    print("Predicting..")
    m, v = pbp.predict(X_test)
    m_squeezed, v_squeezed = tf.squeeze(m), tf.squeeze(v)

    rmse = np.sqrt(np.mean((y_test - m_squeezed) ** 2))
    print(f"{rmse=}")

    # We compute the test log-likelihood
    test_ll = np.mean(
        -0.5 * np.log(2 * math.pi * v_squeezed)
        - 0.5 * (y_test - m_squeezed) ** 2 / v_squeezed
    )
    print(f"{test_ll=}")

    print("Plotting..")
    id = np.arange(X_test.shape[0])
    plt.figure(figsize=(15, 15))
    plt.plot(
        id, scale_y.inverse_transform(y_test), linestyle="", marker=".", label="data"
    )
    plt.plot(id, scale_y.inverse_transform(m), alpha=0.5, label="predict mean")
    plt.fill_between(
        id,
        scale_y.inverse_transform(m + tf.sqrt(v)).squeeze(),
        scale_y.inverse_transform(m - tf.sqrt(v)).squeeze(),
        alpha=0.5,
        label="credible interval",
    )
    plt.xlabel("data id")
    plt.ylabel("target")
    plt.legend()

    plt.savefig(Path(__file__).parent / "pbp_results.png")


if __name__ == "__main__":
    main()
