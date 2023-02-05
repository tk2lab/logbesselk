import math

import tensorflow as tf

from .math import (
    fabs,
    fround,
    log,
    square,
)
from .misc import (
    log_bessel_recurrence,
)
from .utils import (
    epsilon,
    result_shape,
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
    I.J. Thompson and A.R. Barnett,
    Modified Bessel function Iv(z) and Kv(z) and real order
    and complex argument, to selected accuracy,
    Computer Physics Communications, 47, 245-257 (1987).
    """
    n = fround(v)
    u = v - n
    log_ku, log_kup1 = log_bessel_ku(u, x)
    return log_bessel_recurrence(log_ku, log_kup1, u, n, x)[0]


def log_bessel_ku(u, x):
    def cond(si, ri, i, bi, ci, di, fp, fi, gi, hi):
        nonzero_update = fabs(di * hi) > eps * fabs(si)
        return tf.math.reduce_any(nonzero_update)

    def body(si, ri, i, bi, ci, di, fp, fi, gi, hi):
        j = i + 1
        aj = square(j - (1 / 2)) - square(u)
        bj = 2 * (x + j)

        cj = 1 / (bj - aj * ci)
        dj = di * (bj * cj - 1)
        rj = ri + dj

        fj = (bi * fi - fp) / aj
        gj = gi * aj / j
        hj = hi + fj * gj
        sj = si + dj * hj
        return sj, rj, j, bj, cj, dj, fi, fj, gj, hj

    shape = result_shape(u, x)
    dtype = result_type(u, x)
    eps = epsilon(dtype)

    i = tf.constant(1, dtype)
    a1 = (1 / 4) - square(u)
    b1 = 2 * x + 2

    c1 = 1 / b1
    d1 = 1 / b1
    r1 = d1

    f0 = tf.zeros(shape, dtype)
    f1 = tf.ones(shape, dtype)
    g1 = a1
    h1 = f1 * g1
    s1 = 1 + d1 * h1

    init = s1, r1, i, b1, c1, d1, f0, f1, g1, h1
    sn, rn, *_ = tf.while_loop(cond, body, init, maximum_iterations=100)

    c = (1 / 2) * math.log((1 / 2) * math.pi)
    log_ku = c - (1 / 2) * log(x) - x - log(sn)
    log_kup1 = log_ku + log(((1 / 2) + u + x - a1 * rn) / x)
    return log_ku, log_kup1
