import jax.lax as lax
import jax.numpy as jnp

from .asymptotic import log_bessel_k_naive as log_k_large_v
from .cfraction import log_bessel_ku as log_ku_large_x
from .misc import log_bessel_recurrence
from .series import log_bessel_ku as log_ku_small_x
from .wrap import wrap_bessel_k_ratio
from .wrap import wrap_bessel_ke
from .wrap import wrap_log_bessel_k

__all__ = [
    "log_bessel_k",
    "bessel_k_ratio",
    "bessel_ke",
]


@wrap_log_bessel_k
def log_bessel_k(v, x):
    """
    Combination of Series, Continued fraction and Asymptotic expansion.
    """

    def small_v_case():
        n = jnp.round(v)
        u = v - n
        lk0, lk1 = lax.cond(
            small_x,
            lambda: log_ku_small_x(u, x),
            lambda: log_ku_large_x(u, x),
        )
        return log_bessel_recurrence(lk0, lk1, u, n, x)[0]

    def large_v_case():
        return log_k_large_v(v, x)

    small_v = v < 25
    small_x = x < 1.6 + (1 / 2) * jnp.log(v + 1)
    return lax.cond(small_v, small_v_case, large_v_case)


def bessel_k_ratio(v, x, d=1):
    return wrap_bessel_k_ratio(log_bessel_k, v, x, d)


def bessel_ke(v, x):
    return wrap_bessel_ke(log_bessel_k, v, x)
