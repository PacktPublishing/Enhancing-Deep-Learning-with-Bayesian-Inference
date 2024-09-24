import numpy as np
import tensorflow as tf
from scipy.spatial.distance import euclidean
from tensorflow.keras import Model, Sequential, layers, optimizers, metrics, losses
import pandas as pd
from sklearn.preprocessing import StandardScaler
import copy

class RLModel():

    def __init__(self, state_size, n_actions, num_epochs=500):
        self.state_size = state_size
        self.n_actions = n_actions
        self.num_epochs = 200
        self.model = Sequential()
        self.model.add(layers.Dense(20, input_dim=self.state_size, activation='relu', name='layer_1'))
        self.model.add(layers.Dense(8, activation='relu', name='layer_2'))
        self.model.add(layers.Dense(1, activation='relu', name='layer_3'))
        self.model.compile(optimizer=optimizers.Adam(),
                      loss=losses.Huber(),
                      metrics=[metrics.RootMeanSquaredError()],)


    def fit(self, X_train, y_train, batch_size=16):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train, epochs=self.num_epochs, verbose=0, batch_size=batch_size)

    def predict(self, state):
        rewards = []
        X = np.zeros((self.n_actions, self.state_size))
        for i in range(self.n_actions):
            X[i] = np.concatenate([state, [i]])
        X = self.scaler.transform(X)
        rewards = self.model.predict(X)
        return np.argmax(rewards)

class RLModelDropout():

    def __init__(self, state_size, n_actions, num_epochs=200, nb_inference=10):
        self.state_size = state_size
        self.n_actions = n_actions
        self.num_epochs = num_epochs
        self.nb_inference = nb_inference
        self.model = Sequential()
        self.model.add(layers.Dense(10, input_dim=self.state_size, activation='relu', name='layer_1'))
        self.model.add(layers.Dropout(0.15))
        self.model.add(layers.Dense(1, activation='relu', name='layer_3'))
        self.model.compile(optimizer=optimizers.Adam(),
                      loss=losses.Huber(),
                      metrics=[metrics.RootMeanSquaredError()],)

        self.proximity_dict = {"proximity": [], "uncertainty": []}


    def fit(self, X_train, y_train, batch_size=16):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train, epochs=self.num_epochs, verbose=0, batch_size=batch_size)


    def predict(self, state, obstacle_proximity, obstacle=False):
        rewards = []
        X = np.zeros((self.n_actions, self.state_size))
        for i in range(self.n_actions):
            X[i] = np.concatenate([state, [i], [obstacle_proximity[i]]])
        X = self.scaler.transform(X)
        rewards, y_std = self.predict_ll_dropout(X)
        # we subtract our standard deviations from our predicted reward values, this way uncertain predictions are penalised
        rewards = rewards - y_std
        best_action = np.argmax(rewards)
        if obstacle:
            self.proximity_dict["proximity"].append(obstacle_proximity[best_action])
            self.proximity_dict["uncertainty"].append(y_std[best_action][0])
        return best_action

    def predict_ll_dropout(self, X):
        ll_pred = [self.model(X, training=True) for _ in range(self.nb_inference)]
        ll_pred = np.stack(ll_pred)
        return ll_pred.mean(axis=0), ll_pred.std(axis=0)

class RLModelDropoutLL():

    def __init__(self, state_size, n_actions, num_epochs=200, nb_inference=10):
        self.state_size = state_size
        self.n_actions = n_actions
        self.num_epochs = num_epochs
        self.nb_inference = nb_inference
        self.basis_func = Sequential()
        self.basis_func.add(layers.Dense(20, input_dim=self.state_size, activation='relu', name='layer_1'))
        self.basis_func.add(layers.Dense(8, activation='relu', name='layer_2'))
        self.basis_func.add(layers.Dense(1, activation='relu', name='layer_3'))
        self.basis_func.compile(optimizer=optimizers.Adam(),
                      loss=losses.Huber(),
                      metrics=[metrics.RootMeanSquaredError()],)

        self.model = Sequential()
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(1, input_dim=8, activation='relu', name='dropout_layer'))
        self.model.compile(optimizer=optimizers.Adam(),
                      loss=losses.Huber(),
                      metrics=[metrics.RootMeanSquaredError()],)

        self.proximity_dict = {"proximity": [], "uncertainty": []}


    def fit(self, X_train, y_train, batch_size=16):
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        self.basis_func.fit(X_train, y_train, epochs=self.num_epochs, verbose=1, batch_size=batch_size)
        self.basis_func_out = Model(inputs=self.basis_func.input,
                                    outputs=self.basis_func.get_layer('layer_2').output)
        self.model.fit(self.basis_func_out.predict(X_train), y_train, epochs=self.num_epochs, verbose=1, batch_size=batch_size)


    def predict(self, state, obstacle_proximity, obstacle=False):
        rewards = []
        X = np.zeros((self.n_actions, self.state_size))
        for i in range(self.n_actions):
            X[i] = np.concatenate([state, [i], [obstacle_proximity[i]]])
        X = self.scaler.transform(X)
        rewards, y_std = self.predict_ll_dropout(X)
        best_action = np.argmax(rewards)
        if obstacle:
            self.proximity_dict["proximity"].append(obstacle_proximity[best_action])
            self.proximity_dict["uncertainty"].append(y_std[best_action][0])
        return best_action

    def predict_ll_dropout(self, X):
        basis_feats = self.basis_func_out(X)
        ll_pred = [self.model(basis_feats, training=True) for _ in range(self.nb_inference)]
        ll_pred = np.stack(ll_pred)
        return ll_pred.mean(axis=0), ll_pred.std(axis=0)

class RLModelEnsemble():

    def __init__(self, state_size, n_actions, num_epochs=200, n_models=5):
        self.state_size = state_size
        self.n_actions = n_actions
        self.num_epochs = num_epochs
        self.n_models = n_models
        self.models = []
        for i in range(self.n_models):
            model = Sequential()
            model.add(layers.Dense(10, input_dim=self.state_size, activation='relu', name='layer_1'))
            model.add(layers.Dense(1, activation='relu', name='layer_3'))
            model.compile(optimizer=optimizers.Adam(),
                          loss=losses.Huber(),
                          metrics=[metrics.RootMeanSquaredError()],)
            self.models.append(model)

        self.proximity_dict = {"proximity": [], "uncertainty": []}


    def fit(self, X_train, y_train, batch_size=16):
        for model in self.models:
            model.fit(tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train), epochs=self.num_epochs, verbose=0, batch_size=batch_size)


    def predict(self, state, obstacle_proximity, obstacle=False):
        rewards = []
        X = np.zeros((self.n_actions, self.state_size))
        for i in range(self.n_actions):
            X[i] = np.concatenate([state, [i], [obstacle_proximity[i]]])
        rewards, y_std = self.predict_ensemble(X)
        best_action = np.argmax(rewards)
        if obstacle:
            self.proximity_dict["proximity"].append(obstacle_proximity[best_action])
            self.proximity_dict["uncertainty"].append(y_std[best_action][0])
        return best_action

    def predict_ensemble(self, X):
        ens_pred = [model.predict(X) for model in self.models]
        ens_pred = np.stack(ens_pred)
        return ens_pred.mean(axis=0), ens_pred.std(axis=0)
