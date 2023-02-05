import tensorflow as tf

from .asymptotic import log_bessel_k as log_k_large_v
from .cfraction import log_bessel_ku as log_ku_large_x
from .math import (
    fround,
    is_finite,
    log,
)
from .misc import (
    log_bessel_recurrence,
)
from .series import log_bessel_ku as log_ku_small_x
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

    def large_v_case():
        v_ = tf.where(large_v, v, tf.constant(0, dtype))
        return log_k_large_v(v_, x)

    def small_v_case():
        def large_x_ku():
            u_ = tf.where(large_x, u, tf.constant(1 / 2, dtype))
            x_ = tf.where(large_x, x, tf.constant(1, dtype))
            return log_ku_large_x(u_, x_)

        def small_x_ku():
            u_ = tf.where(small_x, u, tf.constant(1 / 2, dtype))
            return log_ku_small_x(u_, x)

        n = fround(v)
        u = v - n
        logk0l, logk1l = large_x_ku()
        logk0s, logk1s = small_x_ku()
        logk0 = tf.where(large_x, logk0l, logk0s)
        logk1 = tf.where(large_x, logk1l, logk1s)
        return log_bessel_recurrence(logk0, logk1, u, n, x)[0]

    dtype = result_type(v, x)
    finite = is_finite(v) & is_finite(x) & (x > 0)
    large_v_ = v >= 25
    large_x_ = x >= 1.6 + (1 / 2) * log(v + 1)

    large_v = finite & large_v_
    large_x = finite & ~large_v_ & large_x_
    small_x = finite & ~large_v_ & ~large_x_
    return tf.where(large_v, large_v_case(), small_v_case())


def bessel_kratio(v, x, d: int = 1):
    return wrap_bessel_kratio(log_bessel_k, v, x, d)


def bessel_ke(v, x):
    return wrap_bessel_ke(log_bessel_k, v, x)
