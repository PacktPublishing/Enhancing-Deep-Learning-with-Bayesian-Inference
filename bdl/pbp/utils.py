import numpy as np
import tensorflow as tf

from bdl.pbp.math import safe_div, safe_exp
from bdl.pbp.pbp_layer import PBPReLULayer, PBPLayer

pi = tf.math.atan(tf.constant(1.0, dtype=tf.float32)) * 4
LOG_INV_SQRT2PI = -0.5 * tf.math.log(2.0 * pi)


def normalize(x, y, output_shape):
    x = ensure_input(x, tf.float32, x.shape[1])
    y = ensure_output(y, tf.float32, output_shape)
    # We normalize the training data to have zero mean and unit standard
    # deviation in the training set if necessary
    mean_X_train, mean_y_train, std_X_train, std_y_train = get_mean_std_x_y(x, y)
    x = (x - np.full(x.shape, mean_X_train)) / np.full(x.shape, std_X_train)
    y = (y - mean_y_train) / std_y_train
    return x, y


def get_mean_std_x_y(x, y):
    std_X_train = np.std(x, 0)
    std_X_train[std_X_train == 0] = 1
    mean_X_train = np.mean(x, 0)
    std_y_train = np.std(y)
    if std_y_train == 0.0:
        std_y_train = 1.0
    mean_y_train = np.mean(y)
    return mean_X_train, mean_y_train, std_X_train, std_y_train


@tf.function
def logZ(diff_square: tf.Tensor, v: tf.Tensor):
    v0 = v + 1e-6
    return tf.reduce_sum(
        -0.5 * (diff_square / v0) + LOG_INV_SQRT2PI - 0.5 * tf.math.log(v0)
    )


@tf.function
def logZ1_minus_logZ2(diff_square: tf.Tensor, v1: tf.Tensor, v2: tf.Tensor):
    """
    log Z1 - log Z2

    Parameters
    ----------
    diff_square : tf.Tensor
        (y - m)^2
    v1 : tf.Tensor
        Z1 = Z(diff_square,v1)
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


def build_layers(last_shape, units, dtype):
    layers = []
    for unit in units[:-1]:
        layer = PBPReLULayer(unit, dtype=dtype)
        layer.build(last_shape)
        layers.append(layer)
        last_shape = unit
    layer = PBPLayer(units[-1], dtype=dtype)
    layer.build(last_shape)
    layers.append(layer)
    return layers


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


def ensure_input(x, dtype, input_shape):
    x = tf.constant(x, dtype=dtype)
    call_rank = tf.rank(tf.constant(0, shape=input_shape, dtype=dtype)) + 1
    if tf.rank(x) < call_rank:
        x = tf.reshape(x, [-1, * input_shape.as_list()])
    return x


def ensure_output(y, dtype, output_dim):
    output_rank = 2
    y = tf.constant(y, dtype=dtype)
    if tf.rank(y) < output_rank:
        y = tf.reshape(y, [-1, output_dim])
    return y
