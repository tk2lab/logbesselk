import jax.lax as lax

from .asymptotic import log_bessel_k_naive as log_k_large_v
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
        v_ = lax.cond(large_v, lambda: v, lambda: dtype(0))
        return log_k_large_v(v_, x)

    def small_v_case():
        def large_x_ku():
            u_ = lax.cond(large_x, lambda: u, lambda: dtype(1 / 2))
            return log_ku_large_x(u_, x)

        def small_x_ku():
            u_, x_ = lax.cond(
                small_x,
                lambda: (u, x),
                lambda: (dtype(1 / 2), dtype(1)),
            )
            return log_ku_small_x(u_, x_)

        n = fround(v)
        u = v - n
        logk0, logk1 = lax.cond(large_x, large_x_ku, small_x_ku)
        return log_bessel_recurrence(logk0, logk1, u, n, x)[0]

    dtype = result_type(v, x)
    finite = is_finite(v) & is_finite(x) & (x > 0)
    large_v_ = v >= 25
    large_x_ = x >= 1.6 + (1 / 2) * log(v + 1)

    large_v = finite & large_v_
    large_x = finite & ~large_v_ & large_x_
    small_x = finite & ~large_v_ & ~large_x_
    return lax.cond(large_v, large_v_case, small_v_case)


def bessel_kratio(v, x, d=1):
    return wrap_bessel_kratio(log_bessel_k, v, x, d)


def bessel_ke(v, x):
    return wrap_bessel_ke(log_bessel_k, v, x)
