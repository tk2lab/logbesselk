import tensorflow as tf
from tensorflow.math import abs, sign
from tensorflow.math import log, log1p, expm1
from tensorflow.math import sinh, cosh, tanh
from tensorflow.math import reduce_logsumexp as log_sum_exp


@tf.custom_gradient
def log_sinh(x):
    def grad(upstream):
        return upstream / tanh(x)
    log2 = log(tf.constant(2, x.dtype))
    return tf.where(x < 20, log(sinh(x)), x - log2), grad


@tf.custom_gradient
def log_cosh(x):
    def grad(upstream):
        return upstream * tanh(x)
    return x + log1p(expm1(-2 * x) / 2), grad


