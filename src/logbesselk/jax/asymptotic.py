import math

import jax.lax as lax
import jax.numpy as jnp

from .math import fabs
from .math import log
from .math import sqrt
from .math import square
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
    p = sqrt(square(v) + square(x))
    q = square(v) / (square(v) + square(x))
    r = (1 / 2) * log(p) + p
    s = v * log((v + p) / x)
    t = calc_sum_fpq(p, q, max_iter=100)
    return c - r + s + log(t)


def calc_sum_fpq(p, q, max_iter):
    def cond(args):
        out, i, faci, pi, diff = args
        return fabs(diff) > eps * fabs(out)

    def body(args):
        out, i, faci, pi, _ = args
        diff = poly(faci, i, q) / pi
        return out + diff, i + 1, update_factor(faci, i), pi * p, diff

    dtype = jnp.result_type(p, q).type
    eps = jnp.finfo(dtype).eps
    outi = dtype(0)
    i = 0
    faci = jnp.asarray([1] + [0] * (max_iter - 1), dtype)
    pi = dtype(1)
    diff = dtype(jnp.inf)
    return lax.while_loop(cond, body, (outi, i, faci, pi, diff))[0]


def update_factor(fac, i):
    k = i + 2 * jnp.arange(fac.size)
    a = ((1 / 2) * k + (1 / 8) / (k + 1)) * fac
    b = ((1 / 2) * k + (5 / 8) / (k + 3)) * fac
    return jnp.pad(b[:-1], [1, 0]) - a


def poly(fac, i, x):
    def cond(args):
        out, j = args
        return j >= 0

    def body(args):
        out, j = args
        return fac[j] + x * out, j - 1

    dtype = jnp.result_type(x).type
    return lax.while_loop(cond, body, (dtype(0), i))[0]
