import os
import pandas as pd
import math
import numpy as np
from ch05.pbp_theano.PBP_net import PBP_net
os.environ['KERAS_BACKEND'] = 'theano'
np.random.seed(1)


def main():
    X_train, y_train, X_test, y_test = get_data()
    net = train(X_train, y_train)
    rmse, test_ll = predict(net, X_test, y_test)
    print(rmse, test_ll)


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


def train(X_train, y_train):
    # We construct the network with one hidden layer with two-hidden layers
    # with 50 neurons in each one and normalizing the training features to have
    # zero mean and unit standard deviation in the trainig set.
    n_hidden_units = 50
    net = PBP_net(
        X_train,
        y_train,
        [n_hidden_units, n_hidden_units],
        normalize=True,
        n_epochs=40
    )
    return net


def predict(net: PBP_net, X_test, y_test):
    # We make predictions for the test set
    m, v, v_noise = net.predict(X_test)
    # We compute the test RMSE
    rmse = np.sqrt(np.mean((y_test - m)**2))

    # We compute the test log-likelihood
    test_ll = np.mean(-0.5 * np.log(2 * math.pi * (v + v_noise)) - \
                      0.5 * (y_test - m)**2 / (v + v_noise))
    return rmse, test_ll


if __name__ == '__main__':
    main()
