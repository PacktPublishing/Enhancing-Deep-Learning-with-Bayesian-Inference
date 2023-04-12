import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers, losses, metrics
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from bdl_base import BDLBase
from constants import DEFAULT_LAYERS

class MCDropout(BDLBase):

    def __init__(self, layers=[]):
        if len(layers) == 0:
            layers = DEFAULT_LAYERS
        self.layers = layers

    def compile_model(self):
        self.model = Sequential()
        for layer in self.layers:
            self.model.add(layer)
        self.model.compile(optimizer=optimizers.Adam(),
                      loss=losses.MeanSquaredError(),
                      metrics=[metrics.RootMeanSquaredError()],)
        self.model.build(input_shape=[None, self.n_features])

    def fit(self, X_train, y_train, batch_size=16, n_epochs=10):
        self.compute_mean_std(X_train, y_train)
        X_train = self.normalize_X(X_train)
        y_train = self.normalize_y(y_train)
        self.n_features = X_train.shape[-1]
        self.compile_model()
        self.model.fit(X_train, y_train, epochs=n_epochs, verbose=1, batch_size=batch_size)

    def predict(self, X, num_inference=5):
        X = self.normalize_X(X)
        predictions = tf.stack(
            [self.model(X, training=True)
             for _ in range(num_inference)],axis=0)
        mean_predictions = tf.reduce_mean(
            predictions,
            axis=0)
        var_predictions = tf.math.reduce_variance(
            predictions,
            axis=0)
        return self.inverse_normalize_y(mean_predictions).numpy(), self.inverse_normalize_y_var(var_predictions).numpy()
