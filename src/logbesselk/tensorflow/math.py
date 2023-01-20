import math

import tensorflow as tf

__all__ = [
    "sinc",
    "sinhc",
    "log_sinh",
    "log_cosh",
]


def sinc(x):
    pix = math.pi * x
    return tf.where(
        tf.equal(x, 0),
        tf.constant(1, x.dtype),
        tf.math.sin(pix) / pix,
    )


def sinhc(x):
    return tf.where(
        tf.equal(x, 0),
        tf.constant(1, x.dtype),
        tf.math.sinh(x) / x,
    )


@tf.custom_gradient
def log_sinh(x):
    def grad(upstream):
        return upstream / tf.math.tanh(x)

    out = tf.where(
        x < 20,
        tf.math.log(tf.math.sinh(x)),
        x - math.log(2),
    )
    return out, grad


@tf.custom_gradient
def log_cosh(x):
    def grad(upstream):
        return upstream * tf.math.tanh(x)

    out = x + tf.math.log1p(tf.math.expm1(-2 * x) / 2)
    return out, grad


def log_add_exp(x, y, sign=None):
    larger = tf.math.maximum(x, y)
    if sign is None:
        sign = 1
    return larger + tf.math.log(
        tf.math.exp(x - larger) + sign * tf.math.exp(y - larger)
    )
