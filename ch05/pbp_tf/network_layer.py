import math

# import theano
# import theano.tensor as T
from typing import Optional

import tensorflow as tf


class NetworkLayer:
    def __init__(
        self, m_w_init, v_w_init, non_linear=True, layer_index: Optional[int] = None
    ):
        # We create the theano variables for the means and variances
        self.m_w = tf.Variable(m_w_init, name="m_w")
        self.v_w = tf.Variable(v_w_init, name="v_w")
        self.w = tf.Variable(m_w_init, name="w")
        # self.m_w = theano.shared(value=m_w_init.astype(theano.config.floatX), name='m_w', borrow=True)
        # self.v_w = theano.shared(value=v_w_init.astype(theano.config.floatX), name='v_w', borrow=True)
        # self.w = theano.shared(value=m_w_init.astype(theano.config.floatX), name='w', borrow=True)

        # We store the type of activation function
        self.non_linear = non_linear

        # We store the number of inputs
        self.n_inputs = float(m_w_init.shape[1])
        if layer_index == 0:
            assert m_w_init.shape[1] == 14
        else:
            assert m_w_init.shape[1] == 51
        # self.n_inputs = theano.shared(float(m_w_init.shape[1]))

    @staticmethod
    @tf.function
    def n_pdf(x):
        return (
            1.0
            / tf.math.sqrt(2 * tf.constant(math.pi, dtype=tf.float64))
            * tf.math.exp(0.5 * x**2)
        )

    @staticmethod
    @tf.function
    def n_cdf(x):
        return 0.5 * (
            1.0 + tf.math.erf(x / tf.math.sqrt(tf.constant(2.0, dtype=tf.float64)))
        )

    @staticmethod
    @tf.function
    def gamma(x):
        return NetworkLayer.n_pdf(x) / NetworkLayer.n_cdf(-x)

    @staticmethod
    def beta(x):
        return NetworkLayer.gamma(x) * (NetworkLayer.gamma(x) - x)

    def output_probabilistic(self, m_w_previous, v_w_previous):
        # import ipdb
        # ipdb.set_trace()
        # We add an additional deterministic input with mean 1 and variance 0
        m_w_previous_with_bias = tf.cast(
            tf.concat([m_w_previous, tf.Variable([1], dtype=tf.float64)], 0), tf.float64
        )
        v_w_previous_with_bias = tf.cast(
            tf.concat([v_w_previous, tf.Variable([0], dtype=tf.float64)], 0), tf.float64
        )
        # We compute the mean and variance after the linear operation
        # import ipdb
        # ipdb.set_trace()

        m_linear = tf.tensordot(self.m_w, m_w_previous_with_bias, 1) / tf.cast(
            tf.math.sqrt(self.n_inputs), dtype=tf.float64
        )
        v_linear = (
            tf.tensordot(self.v_w, v_w_previous_with_bias, 1)
            + tf.tensordot(self.m_w**2, v_w_previous_with_bias, 1)
            + tf.tensordot(self.v_w, m_w_previous_with_bias**2, 1)
        ) / self.n_inputs

        if self.non_linear:
            # We compute the mean and variance after the ReLU activation
            alpha = m_linear / tf.math.sqrt(v_linear)
            gamma = NetworkLayer.gamma(-alpha)
            gamma_robust = -alpha - 1.0 / alpha + 2.0 / alpha**3
            gamma_final = tf.where(
                -alpha > tf.cast(tf.fill(alpha.shape, 30.0), tf.float64),
                tf.cast(gamma, tf.float64),
                tf.cast(gamma_robust, tf.float64),
            )
            # gamma_final = T.switch(T.lt(-alpha, T.fill(alpha, 30)), gamma, gamma_robust)

            v_aux = m_linear + tf.math.sqrt(v_linear) * gamma_final
            m_a = NetworkLayer.n_cdf(alpha) * v_aux
            v_a = m_a * v_aux * NetworkLayer.n_cdf(-alpha) + NetworkLayer.n_cdf(
                alpha
            ) * v_linear * (1 - gamma_final * (gamma_final + alpha))
            return m_a, v_a

        else:
            return m_linear, v_linear

    def output_deterministic(self, output_previous):
        # We add an additional input with value 1
        output_previous_with_bias = tf.concat(
            [output_previous, tf.constant([1.0, 1.0])], 0
        ) / tf.math.sqrt(self.n_inputs)

        # We compute the mean and variance after the linear operation
        a = tf.tensordot(self.w, output_previous_with_bias)

        if self.non_linear:
            # We compute the ReLU activation
            a = tf.nn.relu(a)
            # a = T.switch(T.lt(a, T.fill(a, 0)), T.fill(a, 0), a)

        return a
