import math

import jax.lax as lax
import jax.numpy as jnp

from .misc import log_bessel_recurrence
from .wrap import wrap_log_bessel_k

__all__ = [
    "log_bessel_k",
]


@wrap_log_bessel_k
def log_bessel_k(v, x):
    """
    I.J. Thompson and A.R. Barnett,
    Modified Bessel function Iv(z) and Kv(z) and real order
    and complex argument, to selected accuracy,
    Computer Physics Communications, 47, 245-257 (1987).
    """
    n = jnp.round(v)
    u = v - n
    log_ku, log_kup1 = log_bessel_ku(u, x)
    return log_bessel_recurrence(log_ku, log_kup1, u, n, x)[0]


def log_bessel_ku(u, x):
    def cond(args):
        si, ri, i, bi, ci, di, fp, fi, gi, hi = args
        return (i < max_iter) * (jnp.abs(di * hi) >= eps * jnp.abs(si))

    def body(args):
        si, ri, i, bi, ci, di, fp, fi, gi, hi = args
        j = i + 1
        aj = jnp.square(j - (1 / 2)) - jnp.square(u)
        bj = 2 * (x + j)

        cj = 1 / (bj - aj * ci)
        dj = di * (bj * cj - 1)
        rj = ri + dj

        fj = (bi * fi - fp) / aj
        gj = gi * aj / j
        hj = hi + fj * gj
        sj = si + dj * hj
        return sj, rj, j, bj, cj, dj, fi, fj, gj, hj

    max_iter = 100

    dtype = jnp.result_type(u, x)
    eps = jnp.finfo(dtype).eps

    i = 1
    a1 = (1 / 4) - jnp.square(u)
    b1 = 2 * x + 2

    c1 = 1 / b1
    d1 = 1 / b1
    r1 = d1

    f0 = jnp.asarray(0, dtype)
    f1 = jnp.asarray(1, dtype)
    g1 = a1
    h1 = f1 * g1
    s1 = 1 + d1 * h1

    init = s1, r1, i, b1, c1, d1, f0, f1, g1, h1
    sn, rn, *_ = lax.while_loop(cond, body, init)
    log_ku = (1 / 2) * jnp.log((1 / 2) * math.pi / x) - x - jnp.log(sn)
    log_kup1 = log_ku + jnp.log(((1 / 2) + u + x - a1 * rn) / x)
    return log_ku, log_kup1
