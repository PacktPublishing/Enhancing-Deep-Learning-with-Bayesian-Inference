import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from bdl_base import BDLBase

NUM_INFERENCES = 7

class BBBClassifier(BDLBase):

    def __init__(self):
        pass

    def define_bayesian_model(self):

        kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(self.num_train_examples, dtype=tf.float32))

        self.model = tf.keras.models.Sequential([
            tfp.layers.Convolution2DReparameterization(
                64, kernel_size=5, padding='SAME',
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(
                pool_size=[2, 2], strides=[2, 2],
                padding='SAME'),
            tf.keras.layers.Flatten(),
            tfp.layers.DenseReparameterization(
                self.num_classes, kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.softmax)
        ])

    def compile_bayesian_model(self):
        # Model compilation.
        optimizer = tf.keras.optimizers.Adam()
        # We use the categorical_crossentropy loss since the MNIST dataset contains
        # ten labels. The Keras API will then automatically add the
        # Kullback-Leibler divergence (contained on the individual layers of
        # the model), to the cross entropy loss, effectively
        # calcuating the (negated) Evidence Lower Bound Loss (ELBO)
        self.model.compile(optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy'], experimental_run_tf_function=False)

        self.model.build(input_shape=[None, 28, 28, 1])

    def fit(self, X_train, y_train, batch_size=16, n_epochs=10):
         self.num_classes = np.max(y_train) + 1
         y_train_one_hot = tf.one_hot(train_labels, self.num_classes)
         self.num_classes = y_train_one_hot.shape[-1]
         self.num_train_examples = X_train.shape[0]
         self.define_bayesian_model()
         self.compile_bayesian_model()
         self.model.fit(X_train, y_train_one_hot, epochs=n_epochs, batch_size=batch_size)

    def predict(self, X, num_inference=10):
        softmax_predictions = tf.stack(
            [self.model.predict(X)
             for _ in range(num_inference)],axis=0)
        mean_predictions = tf.reduce_mean(
            tf.argmax(softmax_predictions, axis=2),
            axis=1)
        var_predictions= tf.reduce_mean(
            tf.math.reduce_variance(softmax_predictions, axis=0),
            axis=1)
        return mean_predictions.numpy(), var_predictions.numpy()


class BBBRegressor(BDLBase):

    def __init__(self):
        pass

    def define_bayesian_model(self):

        kl_divergence_function = (lambda q, p, _: tfp.distributions.kl_divergence(q, p) / tf.cast(self.num_train_examples, dtype=tf.float32))

        self.model = tf.keras.models.Sequential([
            tfp.layers.DenseReparameterization(
                64,
                kernel_divergence_fn=kl_divergence_function,
                activation=tf.nn.relu),
            tfp.layers.DenseReparameterization(
                1, kernel_divergence_fn=kl_divergence_function,
                activation=tf.keras.activations.linear)
        ])

    def compile_bayesian_model(self):
        # Model compilation.
        optimizer = tf.keras.optimizers.Adam()
        self.model.compile(optimizer, loss='mean_squared_error',
                      metrics=['mean_squared_error'], experimental_run_tf_function=False)

        self.model.build(input_shape=[None, self.n_features])

    def fit(self, X_train, y_train, batch_size=16, n_epochs=20):
        self.compute_mean_std(X_train, y_train)
        X_train = self.normalize_X(X_train)
        y_train = self.normalize_y(y_train)
        self.n_features = X_train.shape[-1]
        self.num_train_examples = X_train.shape[0]
        self.define_bayesian_model()
        self.compile_bayesian_model()
        self.model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size)

    def predict(self, X, num_inference=5):
        X = self.normalize_X(X)
        predictions = tf.stack(
            [self.model.predict(X)
             for _ in range(num_inference)],axis=0)
        mean_predictions = tf.reduce_mean(
            predictions,
            axis=0)
        var_predictions = tf.math.reduce_variance(
            predictions,
            axis=0)
        return self.inverse_normalize_y(mean_predictions).numpy(), self.inverse_normalize_y_var(var_predictions).numpy()

if __name__ == "__main__":
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # set clas names
    CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # bbb = BBBClassifier()
    #
    # bbb.fit(train_images, train_labels)
    # y_pred, y_var = bbb.predict(test_images)
    #
    # accuracy = accuracy_score(test_images, y_pred)


    bbb = BBBRegressor()
    bbb.fit(X_train, y_train)
    y_pred, y_var = bbb.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    #
    # predictions = tf.stack(
    #     [bbb.model.predict(X_test)
    #      for _ in range(50)],axis=0)
    #
    #
    # mean_predictions = tf.reduce_mean(
    #     predictions,
    #     axis=0)
    # var_predictions= tf.reduce_mean(
    #     tf.math.reduce_variance(predictions, axis=0),
    #     axis=1)
