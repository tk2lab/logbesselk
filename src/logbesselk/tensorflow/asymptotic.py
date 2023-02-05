import math

import tensorflow as tf

from .math import (
    fabs,
    log,
    sqrt,
    square,
)
from .utils import (
    epsilon,
    result_shape,
    result_type,
)
from .wrap import (
    wrap_log_bessel_k,
)

__all__ = [
    "log_bessel_k",
]


@wrap_log_bessel_k
def log_bessel_k(v, x):
    """
    Digital Library of Mathematical Functions: https://dlmf.nist.gov/10.41
    """
    return log_bessel_k_naive(v, x)


def log_bessel_k_naive(v, x):
    c = (1 / 2) * math.log((1 / 2) * math.pi)
    p = sqrt(square(v) + square(x))
    q = square(v) / (square(v) + square(x))
    r = (1 / 2) * log(p) + p
    s = v * log((v + p) / x)
    t = calc_sum_fpq(p, q, max_iter=100)
    return c - r + s + log(t)


def calc_sum_fpq(p, q, max_iter):
    def cond(out, fac, p, i, update):
        return tf.math.reduce_any(update)

    def body(outi, faci, pi, i, update):
        diff = poly(faci, i, q) / pi
        outj = outi + diff
        outj = tf.where(update, outj, outi)
        facj = update_factor(faci, i, max_iter)
        pj = pi * p
        j = i + 1
        update &= fabs(diff) > eps * fabs(outj)
        return outj, facj, pj, j, update

    shape = result_shape(p, q)
    dtype = result_type(p, q)
    eps = epsilon(dtype)
    out0 = tf.zeros(shape, dtype)
    fac0 = tf.constant([1] + [0] * (max_iter - 1), dtype)
    p0 = tf.ones(shape, dtype)
    index = tf.constant(0, tf.int32)
    update = tf.ones(shape, tf.bool)
    init = out0, fac0, p0, index, update
    return tf.while_loop(cond, body, init, maximum_iterations=max_iter)[0]


def update_factor(fac, i, size):
    k = i + 2 * tf.range(size)
    k = tf.cast(k, fac.dtype)
    a = ((1 / 2) * k + (1 / 8) / (k + 1)) * fac
    b = ((1 / 2) * k + (5 / 8) / (k + 3)) * fac
    return tf.pad(b[:-1], [[1, 0]]) - a


def poly(fac, i, x):
    def cond(out, j):
        return j >= 0

    def body(out, j):
        return fac[j] + x * out, j - 1

    return tf.while_loop(cond, body, (tf.zeros_like(x), i))[0]
