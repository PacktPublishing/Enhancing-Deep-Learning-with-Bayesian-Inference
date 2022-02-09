from typing import Iterable, Union, List, Tuple
import tensorflow as tf
tf.random.set_seed(0)
import tensorflow_probability as tfp
import numpy as np
np.random.seed(0)
from ch05.pbp_b4tf.model_base import ModelBase
from IPython import get_ipython
if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from ch05.pbp_b4tf.pbp_layer import PBPReLULayer, PBPLayer

from ch05.pbp_b4tf.utils import safe_div, safe_exp


class PBP(ModelBase):
    """
    Probabilistic Backpropagation

    References
    ----------
    J. M. Hernández-Lobato and R. P. Adams,
    "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks",
    arXiv 1502.05336, 2015
    """

    def __init__(
        self,
        units: List[int],
        *,
        input_shape: Tuple[int] = (1,),
        dtype: Union[tf.dtypes.DType, np.dtype, str] = tf.float32
    ):
        """
        Initialize PBP model

        Parameters
        ----------
        units : Iterable[int]
            Numbers of hidden units and outputs
        input_shape : Iterable[int], optional
            Input shape for PBP model. The default value is `(1,)`
        dtype : tf.dtypes.DType or np.dtype or str
            Data type
        """
        super().__init__(dtype, input_shape, units[-1])
        # hyper-prior for λ - precision parameter.
        self.alpha = tf.Variable(6.0, trainable=True, dtype=self.dtype)
        self.beta = tf.Variable(6.0, trainable=True, dtype=self.dtype)

        pi = tf.math.atan(tf.constant(1.0, dtype=self.dtype)) * 4
        self.log_inv_sqrt2pi = -0.5 * tf.math.log(2.0 * pi)

        last_shape = self.input_shape
        self.layers = []
        for u in units[:-1]:
            # Hidden Layer's Activation is ReLU
            l = PBPReLULayer(u, dtype=self.dtype)
            l.build(last_shape)
            self.layers.append(l)
            last_shape = u

        # Output Layer's Activation is Linear
        l = PBPLayer(units[-1], dtype=self.dtype)
        l.build(last_shape)
        self.layers.append(l)

        self.weights_per_layer = [l.trainable_weights for l in self.layers]

        self.Normal = tfp.distributions.Normal(
            loc=tf.constant(0.0, dtype=self.dtype),
            scale=tf.constant(1.0, dtype=self.dtype),
        )
        self.Gamma = tfp.distributions.Gamma(concentration=self.alpha, rate=self.beta)

    @tf.function
    def _logZ(self, diff_square: tf.Tensor, v: tf.Tensor):
        v0 = v + 1e-6

        return tf.reduce_sum(
            -0.5 * (diff_square / v0) + self.log_inv_sqrt2pi - 0.5 * tf.math.log(v0)
        )

    @tf.function
    def _logZ1_minus_logZ2(self, diff_square: tf.Tensor, v1: tf.Tensor, v2: tf.Tensor):
        """
        log Z1 - log Z2

        Parameters
        ----------
        diff_square : tf.Tensor
            (y - m)^2
        v1 : tf.Tensor
            Z1 = Z(diff_squzre,v1)
        v2 : tf.Tensor
            Z2 = Z(diff_square,v2)

        Returns
        -------
        : tf.Tensor
            log Z1 - log Z2


        Notes
        -----
        diff_square >= 0
        v1 >= 0
        v2 >= 0
        """
        return tf.reduce_sum(
            -0.5 * diff_square * safe_div(v2 - v1, v1 * v2)
            - 0.5 * tf.math.log(safe_div(v1, v2) + 1e-6)
        )

    def fit(self, x, y, batch_size: int = 8, n_epochs: int = 1):
        """
        Fit posterior distribution with observation

        Parameters
        ----------
        x : array-like
            Observed input
        y : array-like
            Observed output
        batch_size : int, optional
            Batch size. The default value is 8.

        Warnings
        --------
        Large batch size might fail because of overflow and/or underflow.
        """
        x = self._ensure_input(x)
        y = self._ensure_output(y)

        data = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

        for _ in tqdm(range(n_epochs), total=n_epochs):
            for _x, _y in data:
                self._fit(_x, _y)

    @tf.function
    def _fit(self, x: tf.Tensor, y: tf.Tensor):
        """Fit on a batch of x and y"""
        with tf.GradientTape() as tape:
            tape.watch(self.weights_per_layer)

            # forward pass through layers
            m, v = self._predict(x)

            # Add beta / (self.alpha - 1) to the variance
            v0 = v + safe_div(self.beta, self.alpha - 1)
            # Squared error
            # JG: Only place where y is used in fit.
            diff_square = tf.math.square(y - m)
            # JG: logarithm of marginal probability of the target variable?
            logZ0 = self._logZ(diff_square, v0)

        # JG: end of first phase?

        grad = tape.gradient(logZ0, self.weights_per_layer)
        for l, g in zip(self.layers, grad):
            l.apply_gradient(g)

        alpha1 = self.alpha + 1
        v1 = v + safe_div(self.beta, self.alpha)
        v2 = v + self.beta / alpha1

        logZ2_logZ1 = self._logZ1_minus_logZ2(diff_square, v1=v2, v2=v1)
        logZ1_logZ0 = self._logZ1_minus_logZ2(diff_square, v1=v1, v2=v0)

        logZ_diff = logZ2_logZ1 - logZ1_logZ0

        Z0Z2_Z1Z1 = safe_exp(logZ_diff)
        # Must update beta first
        # Extract larger exponential
        Pos_where = safe_exp(logZ2_logZ1) * (alpha1 - safe_exp(-logZ_diff) * self.alpha)
        Neg_where = safe_exp(logZ1_logZ0) * (Z0Z2_Z1Z1 * alpha1 - self.alpha)

        beta_denomi = tf.where(logZ_diff >= 0, Pos_where, Neg_where)
        self.beta.assign(
            safe_div(
                self.beta,
                tf.maximum(beta_denomi, tf.zeros_like(self.beta)))
        )

        alpha_denomi = Z0Z2_Z1Z1 * safe_div(alpha1, self.alpha) - 1.0
        self.alpha.assign(
            safe_div(
                tf.constant(1.0, dtype=alpha_denomi.dtype),
                tf.maximum(alpha_denomi, tf.zeros_like(self.alpha)),
            )
        )

    def __call__(self, x):
        """
        Calculate deterministic output

        Parameters
        ----------
        x : array-like
            Input

        Returns
        -------
        y : tf.Tensor
            Neural netork output
        """
        x = self._ensure_input(x)
        return self._call(x)

    @tf.function
    def _call(self, x: tf.Tensor):
        for l in self.layers:
            x = l(x)

        return x + safe_div(
            self.Normal.sample(x.shape), tf.math.sqrt(self.Gamma.sample(x.shape))
        )

    def predict(self, x):
        """
        Predict distribution

        Parameters
        ----------
        x : array-like
            Input

        Returns
        -------
        m : tf.Tensor
            Mean
        v : tf.Tensor
            Variance
        """
        x = self._ensure_input(x)
        m, v = self._predict(x)

        return m, v + safe_div(self.beta, self.alpha - 1)

    @tf.function
    def _predict(self, x: tf.Tensor):
        m, v = x, tf.zeros_like(x)
        for l in self.layers:
            m, v = l.predict(m, v)

        return m, v
