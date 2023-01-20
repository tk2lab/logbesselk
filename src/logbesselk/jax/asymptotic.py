import math

import jax.lax as lax
import jax.numpy as jnp

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
    p = jnp.hypot(v, x)
    q = jnp.square(v / p)
    r = (1 / 2) * jnp.log(p) + p
    s = v * jnp.log((v + p) / x)
    t = calc_sum_fpq(p, q, max_iter=100)
    return c - r + s + jnp.log(t)


def calc_sum_fpq(p, q, max_iter):
    def cond(args):
        out, i, faci, pi, diff = args
        return jnp.abs(diff) > eps * jnp.abs(out)

    def body(args):
        out, i, faci, pi, _ = args
        diff = poly(faci, i + 1, q) / pi
        return out + diff, i + 1, update_factor(faci, i), pi * p, diff

    dtype = jnp.result_type(p, q).type
    eps = jnp.finfo(dtype).eps
    zero = dtype(0)
    one = dtype(1)
    faci = jnp.asarray([1] + [0] * (max_iter - 1), dtype)
    return lax.while_loop(cond, body, (zero, 0, faci, one, one))[0]


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

    dtype = jnp.result_type(x)
    zero = jnp.zeros((), dtype)
    return lax.while_loop(cond, body, (zero, size - 1))[0]
