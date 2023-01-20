import tensorflow as tf

from .math import log
from .math import log_add_exp
from .utils import result_type

__all__ = [
    "sign_bessel_recurrence",
    "log_bessel_recurrence",
]


def sign_deriv_bessel_k(v, x, m=0, n=0):
    dtype = result_type(v, x, tf.float32)
    sign = tf.constant(1, dtype)
    if n % 2 == 1:
        sign *= -1
    if m % 2 == 0:
        return sign
    else:
        return sign(v) * sign


def log_bessel_recurrence(log_ku, log_kup1, u, n, x):
    def cond(ki, kj, ui, ni):
        should_update = ni > 0
        return tf.reduce_any(should_update)

    def body(ki, kj, ui, ni):
        uj = ui + 1
        nj = ni - 1
        kk = log_add_exp(ki, kj + log(2 * uj / x))
        k0 = tf.where(ni > 0, kj, ki)
        k1 = tf.where(ni > 0, kk, kj)
        return k0, k1, uj, nj

    init = log_ku, log_kup1, u, n
    return tf.while_loop(cond, body, init)[:2]
