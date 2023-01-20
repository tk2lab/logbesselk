import math

import tensorflow as tf

from .utils import epsilon
from .utils import result_shape
from .utils import result_type
from .wrap import wrap_log_bessel_k

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
    p = tf.math.sqrt(tf.math.square(v) + tf.math.square(x))
    q = tf.math.square(v) / (tf.math.square(v) + tf.math.square(x))
    r = (1 / 2) * tf.math.log(p) + p
    s = v * tf.math.log((v + p) / x)
    t = calc_sum_fpq(p, q, max_iter=100)
    return c - r + s + tf.math.log(r)


def calc_sum_fpq(p, q, max_iter):
    def cond(out, i, faci, pi, update):
        return tf.math.reduce_any(update)

    def body(outi, i, faci, pi, update):
        j = i + 1
        diff = poly(faci, j, q) / pi
        outj = tf.where(update, outi + diff, outi)
        facj = update_factor(faci, i)
        pj = pi * p
        update = tf.math.abs(diff) > eps * tf.math.abs(outj)
        return outj, j, facj, pj, update

    shape = result_shape(p, q)
    dtype = result_type(p, q)
    eps = epsilon(dtype)
    out0 = tf.zeros(shape, dtype)
    fac0 = tf.constant([1] + [0] * (max_iter - 1), dtype)
    p0 = tf.ones(shape, dtype)
    update = tf.ones(shape, tf.bool)
    return tf.while_loop(cond, body, (out0, 0, fac0, p0, update))[0]


def update_factor(fac, i):
    k = tf.cast(i + 2 * tf.range(tf.size(fac)), fac.dtype)
    shift_fac = tf.pad(fac[1:], [[1, 0]])
    a = ((1 / 2) * k + (1 / 8) / (k + 1)) * fac
    b = ((1 / 2) * k + (5 / 8) / (k + 3)) * shift_fac
    return b - a


def poly(fac, size, x):
    def cond(out, j):
        return j >= 0

    def body(out, j):
        return fac[j] + x * out, j - 1

    init = tf.zeros_like(x), size - 1
    return tf.while_loop(cond, body, init)[0]
