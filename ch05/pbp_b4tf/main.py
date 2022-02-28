import math
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pbp import PBP


NUM_EPOCHS = 40
RANDOM_SEED = 3
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def main():
    print("Get data..")
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = get_data()

    print("Fit..")
    model = fit(X_train, y_train, n_epochs=NUM_EPOCHS)

    print("Predict..")
    m, v, rmse = predict(model, X_test, y_test, x_scaler, y_scaler)

    print("Plot..")
    plot(X_test, y_test, m, v, NUM_EPOCHS, rmse)


def get_data(normalise_train: bool = False, normalise_test: bool = False):
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


def fit(X_train, y_train, n_epochs: int = 1):
    """Fit model with pbp."""
    pbp = PBP([50, 50, 1], input_shape=X_train.shape[1])
    pbp.fit(X_train, y_train, batch_size=1, n_epochs=n_epochs, normalize=True)
    return pbp


def predict(pbp, X_test, y_test, x_scaler, y_scaler):
    # perform inference on normalised data
    m, v, v_noise = pbp.predict_theanolike(X_test)

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
    return m, v, rmse


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
    plt.savefig(Path(__file__).parent / f"pbp_results_tf_{num_epochs:02}_{RANDOM_SEED}.png")


if __name__ == "__main__":
    main()
