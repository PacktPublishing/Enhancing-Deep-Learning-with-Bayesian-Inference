import tensorflow as tf
from tensorflow.python.framework import tensor_shape

import tensorflow_probability as tfp

from bdl.ch05.pbp_b4tf.gamma_initializer import ReciprocalGammaInitializer
from bdl.ch05.pbp_b4tf.utils import safe_div, non_negative_constraint


class PBPLayer(tf.keras.layers.Layer):
    """
    Layer for Probabilistic Backpropagation
    """

    def __init__(self, units: int, dtype=tf.float32, *args, **kwargs):
        """
        Initialize PBP layer

        Parameters
        ----------
        units: int
           Number of units in layer. (Output shape)
        dtype : tf.dtypes.DType
           Data type
        """
        super().__init__(dtype=tf.as_dtype(dtype), *args, **kwargs)
        self.units = units

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to `PBPLayer` "
                "should be defined. Found `None`."
            )
        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.inv_sqrtV1 = tf.cast(
            1.0 / tf.math.sqrt(1.0 * last_dim + 1), dtype=self.dtype
        )
        self.inv_V1 = tf.math.square(self.inv_sqrtV1)

        over_gamma = ReciprocalGammaInitializer(6.0, 6.0)

        # TODO:: Why HeNormal normalization? Not clear that that's done in paper
        # initializers.ones() gives closer rmse (8.30 vs. 6.65, theano 8.18),
        # but plot looks much worse.
        # Theano code seems to
        self.kernel_m = self.add_weight(
            "kernel_mean",
            shape=[last_dim, self.units],
            initializer=tf.keras.initializers.ones(),
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
            shape=[
                self.units,
            ],
            initializer=tf.keras.initializers.zeros(),
            dtype=self.dtype,
            trainable=True,
        )
        self.bias_v = self.add_weight(
            "bias_variance",
            shape=[
                self.units,
            ],
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
        """
        Applay gradient and update weights and bias

        Parameters
        ----------
        gradient : list
            List of gradients for weights and bias.
            [d(logZ)/d(kernel_m), d(logZ)/d(kernel_v), d(logZ)/d(bias_m), d(logZ)/d(bias_v)]
        """

        dlogZ_dkm, dlogZ_dkv, dlogZ_dbm, dlogZ_dbv = gradient

        # Kernel
        self.kernel_m.assign_add(self.kernel_v * dlogZ_dkm)
        new_kv = self.kernel_v - (
            tf.math.square(self.kernel_v) * (tf.math.square(dlogZ_dkm) - 2 * dlogZ_dkv)
        )
        self.kernel_v.assign(non_negative_constraint(new_kv))

        # Bias
        self.bias_m.assign_add(self.bias_v * dlogZ_dbm)
        new_bv = self.bias_v - (
            tf.math.square(self.bias_v) * (tf.math.square(dlogZ_dbm) - 2 * dlogZ_dbv)
        )
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
        """
        Calculate deterministic output

        Parameters
        ----------
        x : tf.Tensor
            Input. [batch, prev_units]

        Returns
        -------
        z : tf.Tensor
            Output. [batch, units]
        """
        W, b = self._sample_weights()
        return (
            tf.tensordot(x, W, axes=[1, 0]) + tf.expand_dims(b, axis=0)
        ) * self.inv_sqrtV1

    def get_config(self):
        return {**super().get_config()}

    @tf.function
    def predict(self, m_prev: tf.Tensor, v_prev: tf.Tensor):
        """
        Predict Mean and Variance

        Parameters
        ----------
        m_prev : tf.Tensor
            Previous Mean. [batch, prev_units]
        v_prev : tf.Tensor
            Previous Variance. [batch, prev_units]

        Returns
        -------
        m : tf.Tensor
            Mean. [batch, units]
        v : tf.Tensor
            Variance. [batch, units]
        """
        # TODO:: potential difference -- no additional input with mean 1 and variance
        # added? See Network_layer line 46
        m = (
            tf.tensordot(m_prev, self.kernel_m, axes=[1, 0])
            + tf.expand_dims(self.bias_m, axis=0)
        ) * self.inv_sqrtV1

        v = (
            tf.tensordot(v_prev, tf.math.square(self.kernel_m), axes=[1, 0])
            + tf.tensordot(tf.math.square(m_prev), self.kernel_v, axes=[1, 0])
            + tf.expand_dims(self.bias_v, axis=0)
            + tf.tensordot(v_prev, self.kernel_v, axes=[1, 0])
        ) * self.inv_V1

        return m, v


class PBPReLULayer(PBPLayer):
    @tf.function
    def call(self, x: tf.Tensor):
        """
        Calculate deterministic output

        Parameters
        ----------
        x : tf.Tensor
            Input. [batch, prev_units]

        Returns
        -------
        z : tf.Tensor
            Output. [batch, units]
        """
        x = super().call(x)
        return tf.maximum(x, tf.zeros_like(x))

    @tf.function
    def predict(self, m_prev: tf.Tensor, v_prev: tf.Tensor):
        """
        Predict Mean and Variance

        Parameters
        ----------
        m_prev : tf.Tensor
            Previous Mean
        v_prev : tf.Tensor
            Previous Variance

        Returns
        -------
        mb : tf.Tensor
            Mean
        vb : tf.Tensor
            Variance
        """
        ma, va = super().predict(m_prev, v_prev)

        _sqrt_v = tf.math.sqrt(tf.maximum(va, tf.zeros_like(va)))
        _alpha = safe_div(ma, _sqrt_v)

        _inv_alpha = safe_div(tf.constant(1.0, dtype=_alpha.dtype), _alpha)

        _cdf_alpha = self.Normal.cdf(_alpha)
        _gamma = tf.where(
            _alpha < -30,
            -_alpha + _inv_alpha * (-1 + 2 * tf.math.square(_inv_alpha)),
            safe_div(self.Normal.prob(-_alpha), _cdf_alpha),
        )
        _vp = ma + _sqrt_v * _gamma

        mb = _cdf_alpha * _vp
        vb = mb * _vp * self.Normal.cdf(-_alpha) + _cdf_alpha * va * (
            1 - _gamma * (_gamma + _alpha)
        )

        return mb, vb
