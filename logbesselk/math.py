import tensorflow as tf
import numpy as np


pi = np.pi
inf = np.inf
nan = np.nan


is_nan = tf.math.is_nan
is_finite = tf.math.is_finite


minimum = tf.math.minimum
maximum = tf.math.maximum

sign = tf.math.sign
abs = tf.math.abs
round = tf.math.round

reciprocal = tf.math.reciprocal
square = tf.math.square
sqrt = tf.math.sqrt
pow = tf.math.pow

log = tf.math.log
log1p = tf.math.log1p
xlogy = tf.math.xlogy

exp = tf.math.exp
expm1 = tf.math.expm1

sin = tf.math.sin
cos = tf.math.cos
tan = tf.math.tan

sinh = tf.math.sinh
cosh = tf.math.cosh
tanh = tf.math.tanh

log_gamma = tf.math.lgamma

sum = tf.math.reduce_sum
log_sum_exp = tf.math.reduce_logsumexp


def epsilon(dtype):
    return np.finfo(dtype.as_numpy_dtype).eps


def gamma(x):
    return exp(log_gamma(x))


@tf.custom_gradient
def log_sinh(x):
    def grad(upstream):
        return upstream / tanh(x)
    return tf.where(x < 20., log(sinh(x)), x - np.log(2.)), grad


@tf.custom_gradient
def log_cosh(x):
    def grad(upstream):
        return upstream * tanh(x)
    return x + log1p(expm1(-2 * x) / 2), grad


def sinc(x):
    pix = np.pi * x
    return tf.where(tf.equal(x, 0.), tf.cast(1., x.dtype), sin(pix) / pix)


def sinhc(x):
    return tf.where(tf.equal(x, 0.), tf.cast(1., x.dtype), sinh(x) / x)


def log_add_exp(x, y, sign=None):
    larger = maximum(x, y)
    if sign is None:
        sign = 1.
    return larger + log(exp(x - larger) + sign * exp(y - larger))


def log_sub_exp(x, y):
    larger = maximum(x, y)
    return larger + log(exp(x - larger) - exp(y - larger))
