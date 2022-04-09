import tensorflow as tf


@tf.function
def safe_div(x: tf.Tensor, y: tf.Tensor, eps: tf.Tensor = tf.constant(1e-6)):
    """
    Non overflow division for positive tf.Tensor

    Parameters
    ----------
    x : tf.Tensor
        Numerator
    y : tf.Tensor
        Denominator
    eps : tf.Tensor, optional
        Small positive value. The default is 1e-6

    Returns
    -------
    sign(y) * x/(|y|+eps) : tf.Tensor
        Results

    Notes
    -----
    User must guaruantee `eps >= 0`
    """
    _eps = tf.cast(eps, dtype=y.dtype)
    return x / (tf.where(y >= 0, y + _eps, y - _eps))


@tf.function
def safe_exp(x: tf.Tensor, BIG: tf.Tensor = tf.constant(20)):
    """
    Non overflow exp(x)

    Parameters
    ----------
    x : tf.Tensor
        Input
    BIG : tf.Tensor, optional
        Maximum exponent. The default value is 20 (exp(x) <= 1e+20).

    Returns
    -------
    exp(min(x,BIG)) : tf.Tensor
        Results
    """
    return tf.math.exp(tf.math.minimum(x, tf.cast(BIG, dtype=x.dtype)))


@tf.function
def non_negative_constraint(x: tf.Tensor):
    return tf.maximum(x, tf.zeros_like(x))
