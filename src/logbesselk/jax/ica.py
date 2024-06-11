import jax.lax as lax

from .asymptotic import log_bessel_k_naive as log_k_large_v
from .cfraction import log_bessel_ku as log_ku_small_v
from .integral import log_abs_deriv_bessel_k as log_k_small_x
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
    Combination of Integrate, Continued fraction and Asymptotic expansion.
    """

    def large_x_case():
        def small_v_case():
            n = fround(v)
            u = (v - n).astype(dtype)
            u_ = lax.cond(small_v, lambda: u, lambda: dtype(1 / 2))
            logk0, logk1 = log_ku_small_v(u_, x)
            return log_bessel_recurrence(logk0, logk1, u, n, x)[0]

        def large_v_case():
            v_ = lax.cond(large_v, lambda: v.astype(dtype), lambda: dtype(0))
            return log_k_large_v(v_, x)

        dtype = result_type(v, x)
        large_v_ = v >= 25
        small_v = finite & ~large_v_
        large_v = finite & large_v_
        return lax.cond(small_v, small_v_case, large_v_case)

    out = log_k_small_x(v, x)
    finite = is_finite(out)
    return lax.cond(finite, lambda: out, large_x_case)


def bessel_kratio(v, x, d=1):
    return wrap_bessel_kratio(log_bessel_k, v, x, d)


def bessel_ke(v, x):
    return wrap_bessel_ke(log_bessel_k, v, x)
