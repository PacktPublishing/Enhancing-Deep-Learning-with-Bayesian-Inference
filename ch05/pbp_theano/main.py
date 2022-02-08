import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from PBP_net import PBP_net
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


os.environ["KERAS_BACKEND"] = "theano"
np.random.seed(1)


def main():
    print("Getting data..")
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = get_data()
    # X_train_old, y_train_old, X_test_old, y_test_old = get_old_data()
    print("Training..")
    # set normalise=False because data has been normalised already
    # squeeze y_train to reduce dims to expected dims for model fitting
    net = train(X_train, np.squeeze(y_train), normalize=False)
    print("Inference..")
    m, v, y_test = predict(net, X_test, y_test)
    print("Plot..")
    plot(X_test, m, v, y_scaler, y_test)


def get_data():
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
    return X_train, y_train, X_test, y_test, x_scaler, y_scaler


def train(X_train, y_train, n_hidden_units: int = 50, normalize: bool=True):
    # We construct the network with one hidden layer with two-hidden layers
    # with 50 neurons in each one and normalizing the training features to have
    # zero mean and unit standard deviation in the trainig set.
    print(f"{normalize=}")
    net = PBP_net(
        X_train, y_train, [n_hidden_units, n_hidden_units], normalize=normalize, n_epochs=40
    )
    return net


def predict(net: PBP_net, X_test, y_test):
    # We make predictions for the test set
    m, v, v_noise = net.predict(X_test)
    # We compute the test RMSE
    rmse = np.sqrt(np.mean((y_test - m) ** 2))
    print(f"{rmse=}")

    # We compute the test log-likelihood
    test_ll = np.mean(
        -0.5 * np.log(2 * math.pi * (v + v_noise))
        - 0.5 * (y_test - m) ** 2 / (v + v_noise)
    )
    print(f"{test_ll=}")

    return m, v, y_test


def plot(X_test, m, v, y_scaler, y_test):
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
    plt.savefig(Path(__file__).parent / "pbp_results_theano.png")


def get_old_data():
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



if __name__ == "__main__":
    main()
