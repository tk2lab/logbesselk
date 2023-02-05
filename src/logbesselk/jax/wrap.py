import functools

import jax
import jax.lax as lax

from .math import (
    exp,
    fabs,
    inf,
    nan,
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
    @functools.wraps(func)
    def wrapped_func(v, x):
        dtype = result_type(v, x)
        v = fabs(v)
        return lax.cond(
            x > 0,
            lambda: func(v, x),
            lambda: lax.cond(
                x < 0,
                lambda: dtype(nan),
                lambda: dtype(inf),
            ),
        )

    def fwd(v, x):
        out = func(v, x)
        return out, (v, x, out)

    def bwd(res, g):
        v, x, out = res
        outm1 = custom_func(v - 1, x)
        dx = -v / x - exp(outm1 - out)
        return None, dx * g

    custom_func = jax.custom_vjp(wrapped_func)
    custom_func.defvjp(fwd, bwd)
    return custom_func


def wrap_log_abs_deriv_bessel_k(func):
    @functools.wraps(func)
    def wrapped_func(v, x, m: int = 0, n: int = 0):
        if m < 0:
            raise ValueError()
        if n < 0:
            raise ValueError()
        dtype = result_type(v, x)
        v = fabs(v)
        return lax.cond(
            (x > 0) if m % 2 == 0 else (x > 0) & (v > 0),
            lambda: func(v, x, m, n),
            lambda: lax.cond(
                x < 0,
                lambda: dtype(nan),
                lambda: lax.cond(
                    x == 0,
                    lambda: dtype(inf),
                    lambda: dtype(-inf),
                ),
            ),
        )

    def fwd(v, x, m, n):
        out = custom_func(v, x, m, n)
        return out, (v, x, out)

    def bwd(m, n, res, g):
        v, x, out = res
        dv = sign(v) * exp(custom_func(v, x, m + 1, n) - out)
        if (m == 0) & (n == 0):
            dx = -v / x - exp(custom_func(v - 1, x) - out)
        else:
            dx = -exp(custom_func(v, x, m, n + 1) - out)
        return dv * g, dx * g

    custom_func = jax.custom_vjp(wrapped_func, nondiff_argnums=(2, 3))
    custom_func.defvjp(fwd, bwd)
    return custom_func


def wrap_bessel_ke(log_bessel_k, v, x):
    logk = log_bessel_k(v, x)
    return exp(logk + x)


def wrap_bessel_kratio(log_bessel_k, v, x, d=1):
    logk = log_bessel_k(v, x)
    logkd = log_bessel_k(v + d, x)
    return exp(logkd - logk)
