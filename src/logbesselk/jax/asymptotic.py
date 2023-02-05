import math

import jax.lax as lax
import jax.numpy as jnp

from .math import (
    fabs,
    log,
    sqrt,
    square,
)
from .utils import (
    epsilon,
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
    def cond(args):
        return args[-1]

    def body(args):
        outi, faci, pi, i, _ = args
        diff = poly(faci, i, q) / pi
        outj = outi + diff
        facj = update_factor(faci, i, max_iter)
        pj = pi * p
        j = i + 1
        update = fabs(diff) > eps * fabs(outj)
        return outj, facj, pj, j, update

    dtype = result_type(p, q)
    eps = epsilon(dtype)
    out0 = dtype(0)
    fac0 = jnp.asarray([1] + [0] * (max_iter - 1), dtype)
    p0 = dtype(1)
    index = 0
    update = True
    init = out0, fac0, p0, index, update
    return lax.while_loop(cond, body, init)[0]


def update_factor(fac, i, size):
    k = i + 2 * jnp.arange(size)
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

    return lax.while_loop(cond, body, (result_type(x)(0), i))[0]
