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
NUM_EPOCHS = 40


def main():
    print("Get data..")
    X_train_norm, y_train_norm, X_test, y_test, x_scaler, y_scaler = get_data(normalise_train=False)
    # X_train_old, y_train_old, X_test_old, y_test_old = get_old_data()

    print("Fit..")
    # set normalise=False because data has been normalised already
    model = fit(X_train_norm, y_train_norm, normalize=True, n_epochs=NUM_EPOCHS)
    # model = fit(X_train_old, y_train_old, normalize=True, n_epochs=NUM_EPOCHS)


    print("Predict..")
    m, v, y_test, rmse = predict(model, X_test, y_test)
    # m, v, y_test = predict(model, X_test_old, y_test_old)

    # print("Plot..")
    plot(X_test, y_test, m, v, NUM_EPOCHS, rmse)


def get_data(normalise_train: bool = True, normalise_test: bool = False):
    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    if normalise_train:
        X_train = x_scaler.fit_transform(X_train)
        y_train= y_scaler.fit_transform(y_train.reshape(-1, 1))
    if normalise_test:
        X_test = x_scaler.fit_transform(X_test)
        y_test= y_scaler.fit_transform(y_test.reshape(-1, 1))
    return X_train, y_train, X_test, y_test, x_scaler, y_scaler


def fit(
    X_train,
    y_train,
    n_hidden_units: int = 50,
    n_epochs: int = 40,
    normalize: bool = False,
):
    # squeeze y_train to reduce dims to expected dims for model fitting
    y_train= np.squeeze(y_train)
    # fit the model
    net = PBP_net(
        X_train,
        y_train,
        [n_hidden_units, n_hidden_units],
        normalize=normalize,
        n_epochs=n_epochs,
    )
    return net


def predict(net: PBP_net, X_test, y_test):
    # perform inference
    m, v, v_noise = net.predict(X_test)
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

    return m, v, y_test, rmse


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
    plt.title(f"Theano, {num_epochs} epochs, rmse: {rmse}")
    plt.legend()
    plt.savefig(Path(__file__).parent / f"pbp_results_theano_{num_epochs:02}.png")


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
