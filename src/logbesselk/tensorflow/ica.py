import tensorflow as tf

from .asymptotic import log_bessel_k as log_k_large_v
from .cfraction import log_bessel_ku as log_ku_small_v
from .integral import log_bessel_k as log_k_small_x
from .math import (
    fround,
    is_finite,
)
from .misc import (
    log_bessel_recurrence,
)
from .utils import (
    result_type,
)
from .wrap import (
    wrap_bessel_ke,
    wrap_bessel_kratio,
    wrap_log_bessel_k,
)

__all__ = [
    "log_bessel_k",
    "bessel_kratio",
    "bessel_ke",
]


@wrap_log_bessel_k
def log_bessel_k(v, x):
    """
    Combination of Series, Continued fraction and Asymptotic expansion.
    """

    def small_x_case():
        return log_k_small_x(v, x)

    def large_v_case():
        v_ = tf.where(large_v, v, tf.constant(0, dtype))
        return log_k_large_v(v_, x)

    def small_v_case():
        n = fround(v)
        u = v - n
        u_ = tf.where(small_v, u, tf.constant(1 / 2, dtype))
        x_ = tf.where(small_v, x, tf.constant(1, dtype))
        logk0, logk1 = log_ku_small_v(u_, x_)
        return log_bessel_recurrence(logk0, logk1, u, n, x)[0]

    dtype = result_type(v, x)
    finite = is_finite(v) & is_finite(x) & (x > 0)
    large_x_ = x >= 100
    large_v_ = v >= 25
    small_v = finite & large_x_ & large_v_
    large_v = finite & large_x_ & ~large_v_
    out = log_k_small_x(v, x)
    out = tf.where(small_v, small_v_case(), out)
    out = tf.where(large_v, large_v_case(), out)
    return out


def bessel_kratio(v, x, d: int = 1):
    return wrap_bessel_kratio(log_bessel_k, v, x, d)


def bessel_ke(v, x):
    return wrap_bessel_ke(log_bessel_k, v, x)
