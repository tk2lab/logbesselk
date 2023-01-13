import math

import jax.lax as lax
import jax.numpy as jnp

from .wrap import wrap_simple

__all__ = [
    "log_bessel_k",
]


def log_bessel_k_naive(v, x):
    """
    Digital Library of Mathematical Functions: https://dlmf.nist.gov/10.41
    """
    c = (1 / 2) * math.log((1 / 2) * math.pi)
    p = jnp.hypot(v, x)
    q = jnp.square(v / p)
    r = calc_r(p, q, max_iter=100)
    return c - (1 / 2) * jnp.log(p) - p + v * jnp.log((v + p) / x) + jnp.log(r)


@wrap_simple
def log_bessel_k(v, x):
    return log_bessel_k_naive(v, x)


def update_factor(fac, i):
    k = i + 2 * jnp.arange(fac.size)
    a = ((1 / 2) * k + (1 / 8) / (k + 1)) * fac
    b = ((1 / 2) * k + (5 / 8) / (k + 3)) * jnp.pad(fac[1:], [1, 0])
    return b - a


def poly(fac, size, x):
    def cond(args):
        out, j = args
        return j >= 0

    def body(args):
        out, j = args
        return fac[j] + x * out, j - 1

    dtype = jnp.result_type(v, x)
    out = jnp.zeros((), dtype)
    return lax.while_loop(cond, body, (out, size - 1))[0]


def calc_r(p, q, max_iter):
    def cond(args):
        out, i, faci, pi, diff = args
        return jnp.abs(diff) > tol * jnp.abs(out)

    def body(args):
        out, i, faci, pi, _ = args
        diff = poly(faci, i + 1, q) / pi
        return out + diff, i + 1, update_factor(faci, i), pi * p, diff

    dtype = jnp.result_type(p, q)
    tol = jnp.finfo(dtype).eps
    one = jnp.ones((), dtype)
    faci = jnp.asarray([1] + [0] * (max_iter - 1), dtype)
    return lax.while_loop(cond, body, (one, 0, faci, one, one))[0]
