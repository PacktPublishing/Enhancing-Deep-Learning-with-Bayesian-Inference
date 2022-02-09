import math
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from pbp import PBP

np.random.seed(0)
tf.random.set_seed(0)
NUM_EPOCHS = 40

def main():
    print("Get data..")
    X_train_norm, y_train_norm, X_test, y_test, x_scaler, y_scaler = get_data()

    print("Fit..")
    model = fit(X_train_norm, y_train_norm, n_epochs=NUM_EPOCHS)
    
    print("Predict..")
    m, v, y_test_norm, rmse = predict(model, X_test, y_test, x_scaler, y_scaler)

    print("Plot..")
    plot(X_test, y_test_norm, m, v, y_scaler, NUM_EPOCHS, rmse)


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


def fit(X_train, y_train, n_epochs: int = 1):
    """Fit model with pbp."""
    pbp = PBP([50, 50, 1], input_shape=X_train.shape[1])
    pbp.fit(X_train, y_train, batch_size=8, n_epochs=n_epochs)
    return pbp


def predict(pbp, X_test, y_test, x_scaler, y_scaler):
    # normalise data
    X_test_norm = x_scaler.fit_transform(X_test)
    y_test_norm = y_scaler.fit_transform(y_test.reshape(-1, 1))
    # perform inference
    m, v = pbp.predict(X_test_norm)
    # calculate rmse
    m_squeezed, v_squeezed = tf.squeeze(m), tf.squeeze(v)
    rmse = np.sqrt(np.mean((y_test_norm - m_squeezed) ** 2))
    print(f"{rmse=}")
    # calculate log-likelihood
    test_ll = np.mean(
        -0.5 * np.log(2 * math.pi * v_squeezed)
        - 0.5 * (y_test_norm - m_squeezed) ** 2 / v_squeezed
    )
    print(f"{test_ll=}")
    return m, v, y_test_norm, rmse


def plot(X_test, y_test_norm, m, v, y_scaler, num_epochs, rmse):
    id = np.arange(X_test.shape[0])
    plt.figure(figsize=(15, 15))
    plt.plot(
        id, y_scaler.inverse_transform(y_test_norm), linestyle="", marker=".", label="data"
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
    plt.ylabel("target: home price")
    plt.ylim([-10, 60])
    plt.title(f"Tensorflow, {num_epochs} epochs, rmse: {rmse}")
    plt.legend()
    plt.savefig(Path(__file__).parent / f"pbp_results_tf_{num_epochs:02}.png")


if __name__ == "__main__":
    main()
