import numpy as np
import tensorflow as tf
from scipy.spatial.distance import euclidean
from tensorflow.keras import Model, Sequential, layers, optimizers, metrics, losses
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import copy

from bdl_base import BDLBase
from constants import DEFAULT_LAYERS

class Ensemble(BDLBase):

    def __init__(self, layers=[], n_models=5):
        if len(layers) == 0:
            layers = DEFAULT_LAYERS
        self.layers = layers
        self.n_models = n_models

    def compile_model(self):
        self.models = []
        for i in range(self.n_models):
            model = Sequential()
            for layer in self.layers:
                model.add(copy.deepcopy(layer))
            model.compile(optimizer=optimizers.Adam(),
                          loss=losses.MeanSquaredError(),
                          metrics=[metrics.MeanSquaredError()],)
            model.build(input_shape=[None, self.n_features])
            self.models.append(model)

    def fit(self, X_train, y_train, batch_size=16, n_epochs=10):
        self.compute_mean_std(X_train, y_train)
        X_train = self.normalize_X(X_train)
        y_train = self.normalize_y(y_train)
        self.n_features = X_train.shape[-1]
        self.compile_model()
        n_samples = int(0.8*len(X_train))
        for model in self.models:
            idxs = np.random.choice(len(X_train), n_samples, replace=False)
            X_train_model = tf.gather(X_train, indices=idxs)
            y_train_model = tf.gather(y_train, indices=idxs)
            model.fit(X_train_model, y_train_model, epochs=n_epochs, verbose=0, batch_size=batch_size)

    def predict(self, X, num_inference=10):
        X = self.normalize_X(X)
        predictions = tf.stack(
            [model(X)
             for model in self.models],axis=0)
        mean_predictions = tf.reduce_mean(
            predictions,
            axis=0)
        var_predictions = tf.math.reduce_variance(
            predictions,
            axis=0)
        return self.inverse_normalize_y(mean_predictions).numpy(), self.inverse_normalize_y_var(var_predictions).numpy()
