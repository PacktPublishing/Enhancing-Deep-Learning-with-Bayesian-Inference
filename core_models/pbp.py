import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow.python.framework import tensor_shape
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from typing import List, Union, Iterable

from bdl_base import BDLBase

pi = tf.math.atan(tf.constant(1.0, dtype=tf.float32)) * 4
LOG_INV_SQRT2PI = -0.5 * tf.math.log(2.0 * pi)

@tf.function
def safe_div(x: tf.Tensor, y: tf.Tensor, eps: tf.Tensor = tf.constant(1e-6)):
    _eps = tf.cast(eps, dtype=y.dtype)
    return x / (tf.where(y >= 0, y + _eps, y - _eps))


@tf.function
def safe_exp(x: tf.Tensor, BIG: tf.Tensor = tf.constant(20)):
    return tf.math.exp(tf.math.minimum(x, tf.cast(BIG, dtype=x.dtype)))


@tf.function
def non_negative_constraint(x: tf.Tensor):
    return tf.maximum(x, tf.zeros_like(x))

def update_alpha_beta(alpha, beta, diff_square, v, v0):
    alpha1 = alpha + 1
    v1 = v + safe_div(beta, alpha)
    v2 = v + beta / alpha1
    logZ2_logZ1 = logZ1_minus_logZ2(diff_square, v1=v2, v2=v1)
    logZ1_logZ0 = logZ1_minus_logZ2(diff_square, v1=v1, v2=v0)
    logZ_diff = logZ2_logZ1 - logZ1_logZ0
    Z0Z2_Z1Z1 = safe_exp(logZ_diff)
    # Must update beta first
    # Extract larger exponential
    pos_where = safe_exp(logZ2_logZ1) * (alpha1 - safe_exp(-logZ_diff) * alpha)
    neg_where = safe_exp(logZ1_logZ0) * (Z0Z2_Z1Z1 * alpha1 - alpha)
    beta_denomi = tf.where(logZ_diff >= 0, pos_where, neg_where)
    beta = safe_div(beta, tf.maximum(beta_denomi, tf.zeros_like(beta)))

    alpha_denomi = Z0Z2_Z1Z1 * safe_div(alpha1, alpha) - 1.0

    alpha = safe_div(
        tf.constant(1.0, dtype=alpha_denomi.dtype),
        tf.maximum(alpha_denomi, tf.zeros_like(alpha)),
    )

    return alpha, beta


@tf.function
def logZ(diff_square: tf.Tensor, v: tf.Tensor):
    v0 = v + 1e-6
    return tf.reduce_sum(
        -0.5 * (diff_square / v0) + LOG_INV_SQRT2PI - 0.5 * tf.math.log(v0)
    )


@tf.function
def logZ1_minus_logZ2(diff_square: tf.Tensor, v1: tf.Tensor, v2: tf.Tensor):
    return tf.reduce_sum(
        -0.5 * diff_square * safe_div(v2 - v1, v1 * v2)
        - 0.5 * tf.math.log(safe_div(v1, v2) + 1e-6)
        )

class PBPLayer(tf.keras.layers.Layer):
    def __init__(self, units: int, dtype=tf.float32, *args, **kwargs):
        super().__init__(dtype=tf.as_dtype(dtype), *args, **kwargs)
        self.units = units

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.inv_sqrtV1 = tf.cast(1.0 / tf.math.sqrt(1.0 * last_dim + 1), dtype=self.dtype)
        self.inv_V1 = tf.math.square(self.inv_sqrtV1)

        over_gamma = ReciprocalGammaInitializer(6.0, 6.0)
        self.kernel_m = self.add_weight(
            "kernel_mean",
            shape=[last_dim, self.units],
            initializer=tf.keras.initializers.HeNormal(),
            dtype=self.dtype,
            trainable=True,
        )
        self.kernel_v = self.add_weight(
            "kernel_variance",
            shape=[last_dim, self.units],
            initializer=over_gamma,
            dtype=self.dtype,
            trainable=True,
        )
        self.bias_m = self.add_weight(
            "bias_mean",
            shape=[self.units],
            initializer=tf.keras.initializers.HeNormal(),
            dtype=self.dtype,
            trainable=True,
        )
        self.bias_v = self.add_weight(
            "bias_variance",
            shape=[self.units],
            initializer=over_gamma,
            dtype=self.dtype,
            trainable=True,
        )
        self.Normal = tfp.distributions.Normal(
            loc=tf.constant(0.0, dtype=self.dtype),
            scale=tf.constant(1.0, dtype=self.dtype),
        )
        self.built = True

    @tf.function
    def apply_gradient(self, gradient):
        dlogZ_dkm, dlogZ_dkv, dlogZ_dbm, dlogZ_dbv = gradient

        # Kernel
        self.kernel_m.assign_add(self.kernel_v * dlogZ_dkm)
        new_kv = self.kernel_v - (tf.math.square(self.kernel_v) * (tf.math.square(dlogZ_dkm) - 2 * dlogZ_dkv))
        self.kernel_v.assign(non_negative_constraint(new_kv))

        # Bias
        self.bias_m.assign_add(self.bias_v * dlogZ_dbm)
        new_bv = self.bias_v - (tf.math.square(self.bias_v) * (tf.math.square(dlogZ_dbm) - 2 * dlogZ_dbv))
        self.bias_v.assign(non_negative_constraint(new_bv))

    @tf.function
    def _sample_weights(self):
        eps_k = self.Normal.sample(self.kernel_m.shape)
        std_k = tf.math.sqrt(tf.maximum(self.kernel_v, tf.zeros_like(self.kernel_v)))
        W = self.kernel_m + std_k * eps_k

        eps_b = self.Normal.sample(self.bias_m.shape)
        std_b = tf.math.sqrt(tf.maximum(self.bias_v, tf.zeros_like(self.bias_v)))
        b = self.bias_m + std_b * eps_b
        return W, b

    @tf.function
    def call(self, x: tf.Tensor):
        W, b = self._sample_weights()
        return (tf.tensordot(x, W, axes=[1, 0]) + tf.expand_dims(b, axis=0)) * self.inv_sqrtV1

    @tf.function
    def predict(self, previous_mean: tf.Tensor, previous_variance: tf.Tensor):
        mean = (
            tf.tensordot(previous_mean, self.kernel_m, axes=[1, 0])
            + tf.expand_dims(self.bias_m, axis=0)
        ) * self.inv_sqrtV1

        variance = (
            tf.tensordot(previous_variance, tf.math.square(self.kernel_m), axes=[1, 0])
            + tf.tensordot(tf.math.square(previous_mean), self.kernel_v, axes=[1, 0])
            + tf.expand_dims(self.bias_v, axis=0)
            + tf.tensordot(previous_variance, self.kernel_v, axes=[1, 0])
        ) * self.inv_V1

        return mean, variance

class ReciprocalGammaInitializer:
    def __init__(self, alpha, beta):
        self.Gamma = tfp.distributions.Gamma(concentration=alpha, rate=beta)

    def __call__(self, shape: Iterable, dtype=None):
        g = 1.0 / self.Gamma.sample(shape)
        if dtype:
            g = tf.cast(g, dtype=dtype)

        return g

class PBPReLULayer(PBPLayer):
    @tf.function
    def call(self, x: tf.Tensor):
        """Calculate deterministic output"""
        # x is of shape [batch, prev_units]
        x = super().call(x)
        z = tf.maximum(x, tf.zeros_like(x))  # [batch, units]
        return z

    @tf.function
    def predict(self, previous_mean: tf.Tensor, previous_variance: tf.Tensor):
        ma, va = super().predict(previous_mean, previous_variance)
        mb, vb = self.get_mb_vb(ma, va, self.Normal)
        return mb, vb

    def get_mb_vb(self, ma, va, normal):
        _sqrt_v = tf.math.sqrt(tf.maximum(va, tf.zeros_like(va)))
        _alpha = safe_div(ma, _sqrt_v)
        _inv_alpha = safe_div(tf.constant(1.0, dtype=_alpha.dtype), _alpha)
        _cdf_alpha = normal.cdf(_alpha)
        _gamma = tf.where(
            _alpha < -30,
            -_alpha + _inv_alpha * (-1 + 2 * tf.math.square(_inv_alpha)),
            safe_div(normal.prob(-_alpha), _cdf_alpha),
        )
        _vp = ma + _sqrt_v * _gamma
        mb = _cdf_alpha * _vp
        vb = mb * _vp * normal.cdf(-_alpha) + _cdf_alpha * va * (
                1 - _gamma * (_gamma + _alpha)
        )
        return mb, vb

class PBP(BDLBase):

    def __init__(
            self,
            layer_units: List[int]=[64, 1],
            dtype: Union[tf.dtypes.DType, np.dtype, str] = tf.float32
        ):
            self.layer_units = layer_units
            self.dtype = dtype

    def fit(self, X_train, y_train, batch_size: int = 16, n_epochs: int = 10):
        self.compute_mean_std(X_train, y_train)
        X_train = self.normalize_X(X_train)
        y_train = self.normalize_y(y_train)
        layers = self.build_layers(X_train.shape[-1], self.layer_units)
        self.alpha = tf.Variable(6.0, trainable=True, dtype=self.dtype)
        self.beta = tf.Variable(6.0, trainable=True, dtype=self.dtype)
        self.layers = layers
        self.Normal = tfp.distributions.Normal(
            loc=tf.constant(0.0, dtype=self.dtype),
            scale=tf.constant(1.0, dtype=self.dtype),
        )
        self.Gamma = tfp.distributions.Gamma(concentration=self.alpha, rate=self.beta)
        data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        for epoch_index in range(n_epochs):
            print(f"{epoch_index=}")
            for X_batch, y_batch in data:
                diff_square, v, v0 = self.update_gradients(X_batch, y_batch)
                alpha, beta = update_alpha_beta(self.alpha, self.beta, diff_square, v, v0)
                self.alpha.assign(alpha)
                self.beta.assign(beta)

    def build_layers(self, last_shape, units):
        layers = []
        for unit in units[:-1]:
            layer = PBPReLULayer(unit)
            layer.build(last_shape)
            layers.append(layer)
            last_shape = unit
        layer = PBPLayer(units[-1])
        layer.build(last_shape)
        layers.append(layer)
        return layers

    def predict(self, X: tf.Tensor):
        X = self.normalize_X(X)
        y_pred, y_var = self.predict_internal(X)
        return self.inverse_normalize_y(y_pred).numpy(), self.inverse_normalize_y_var(y_var).numpy()

    @tf.function
    def predict_internal(self, X: tf.Tensor):
        mean_predictions, var_predictions = X, tf.zeros_like(X)
        for layer in self.layers:
            mean_predictions, var_predictions = layer.predict(mean_predictions, var_predictions)
        return mean_predictions, var_predictions

    @tf.function
    def update_gradients(self, x, y):
        trainables = [layer.trainable_weights for layer in self.layers]
        with tf.GradientTape() as tape:
            tape.watch(trainables)
            m, v = self.predict_internal(x)
            v0 = v + safe_div(self.beta, self.alpha - 1)
            diff_square = tf.math.square(y - m)
            logZ0 = logZ(diff_square, v0)
        grad = tape.gradient(logZ0, trainables)
        for l, g in zip(self.layers, grad):
            l.apply_gradient(g)
        return diff_square, v, v0
