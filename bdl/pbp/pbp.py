"""
Probabilistic Backpropagation

References
----------
J. M. Hernández-Lobato and R. P. Adams,
"Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks",
arXiv 1502.05336, 2015
"""
from typing import Union, List
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from bdl.pbp.math import safe_div
from bdl.pbp.utils import logZ, update_alpha_beta

RANDOM_SEED = 0
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class PBP:
    def __init__(
        self,
        layers: List[tf.keras.layers.Layer],
        dtype: Union[tf.dtypes.DType, np.dtype, str] = tf.float32
    ):
        self.alpha = tf.Variable(6.0, trainable=True, dtype=dtype)
        self.beta = tf.Variable(6.0, trainable=True, dtype=dtype)
        self.layers = layers
        self.Normal = tfp.distributions.Normal(
            loc=tf.constant(0.0, dtype=dtype),
            scale=tf.constant(1.0, dtype=dtype),
        )
        self.Gamma = tfp.distributions.Gamma(concentration=self.alpha, rate=self.beta)

    def fit(self, x, y, batch_size: int = 16, n_epochs: int = 1):
        data = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        for epoch_index in range(n_epochs):
            print(f"{epoch_index=}")
            for x_batch, y_batch in data:
                diff_square, v, v0 = self.update_gradients(x_batch, y_batch)
                alpha, beta = update_alpha_beta(self.alpha, self.beta, diff_square, v, v0)
                self.alpha.assign(alpha)
                self.beta.assign(beta)

    @tf.function
    def predict(self, x: tf.Tensor):
        m, v = x, tf.zeros_like(x)
        for layer in self.layers:
            m, v = layer.predict(m, v)
        return m, v

    @tf.function
    def update_gradients(self, x, y):
        trainables = [layer.trainable_weights for layer in self.layers]
        with tf.GradientTape() as tape:
            tape.watch(trainables)
            m, v = self.predict(x)
            v0 = v + safe_div(self.beta, self.alpha - 1)
            diff_square = tf.math.square(y - m)
            logZ0 = logZ(diff_square, v0)
        grad = tape.gradient(logZ0, trainables)
        for l, g in zip(self.layers, grad):
            l.apply_gradient(g)
        return diff_square, v, v0
