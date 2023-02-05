import functools

import numpy as np
import tensorflow as tf

from .math import (
    exp,
    fabs,
    sign,
)
from .utils import (
    result_type,
)

__all__ = [
    "wrap_log_bessel_k",
    "wrap_log_abs_deriv_bessel_k",
    "wrap_bessel_ke",
    "wrap_bessel_kratio",
]


def wrap_log_bessel_k(func):
    @tf.custom_gradient
    @functools.wraps(func)
    def wrapped_func(v, x):
        dtype = result_type(v, x)
        v = tf.convert_to_tensor(v)
        x = tf.convert_to_tensor(x)
        nan = tf.constant(np.nan, dtype)
        inf = tf.constant(np.inf, dtype)
        v = fabs(v)
        logk = func(v, x)
        logk = tf.where(x > 0, logk, tf.where(x < 0, nan, inf))

        def grad(upstream):
            logk = wrapped_func(v, x)
            logkm1 = wrapped_func(v - 1, x)
            dx = -v / x - exp(logkm1 - logk)
            return None, dx * upstream

        return logk, grad

    return wrapped_func


def wrap_log_abs_deriv_bessel_k(func):
    @tf.custom_gradient
    @functools.wraps(func)
    def wrapped_func(v, x, m=0, n=0):
        dtype = result_type(v, x)
        v = tf.convert_to_tensor(v)
        x = tf.convert_to_tensor(x)
        nan = tf.constant(np.nan, dtype)
        inf = tf.constant(np.inf, dtype)
        v = fabs(v)
        is_finite = tf.where(tf.equal(m % 2, 0), (x > 0), (x > 0) & (v > 0))
        infval = tf.where(x < 0, nan, tf.where(tf.equal(x, 0), inf, -inf))
        logk = func(v, x, m, n)
        logk = tf.where(is_finite, logk, infval)

        def grad(g):
            sv = sign(v)
            logk = wrapped_func(v, x, m, n)
            dv = sv * exp(wrapped_func(v, x, m + 1, n) - logk)
            dx = tf.where(
                tf.equal(m, 0) & tf.equal(n, 0),
                -v / x - exp(wrapped_func(v - 1, x) - logk),
                -exp(wrapped_func(v, x, m, n + 1) - logk),
            )
            return dv * g, dx * g

        return logk, grad

    return wrapped_func


def wrap_bessel_ke(log_bessel_k, v, x):
    logk = log_bessel_k(v, x)
    return exp(logk + x)


def wrap_bessel_kratio(log_bessel_k, v, x, d=1):
    logk = log_bessel_k(v, x)
    logkd = log_bessel_k(v + d, x)
    return exp(logkd - logk)
