import tensorflow as tf

from . import math as tk
from .utils import wrap_log_k
from .utils import log_bessel_recurrence


@wrap_log_k
def log_bessel_k(v, x):
    log_ku, log_kup1 = log_ku(v, x)
    return _log_bessel_recurrence(log_ku, log_kup1, u, n, x)[0]


def _log_bessel_ku(v, x, mask=None):
    """
    I.J. Thompson and A.R. Barnett,
    Modified Bessel function Iv(z) and Kv(z) and real order
    and complex argument, to selected accuracy,
    Computer Physics Communications, 47, 245-257 (1987).
    """

    def cond(si, ri, i, bi, ci, di, fp, fi, gi, hi):
        nonzero_update = tk.abs(di * hi) > tol * tk.abs(si)
        if mask is not None:
            nonzero_update &= mask
        return tf.reduce_any(nonzero_update)

    def body(si, ri, i, bi, ci, di, fp, fi, gi, hi):
        j = i + 1.
        aj = tf.square(j - 0.5) - tf.square(u)
        bj = 2. * (x + j)

        cj = 1. / (bj - aj * ci)
        dj = di * (bj * cj - 1.)
        rj = ri + dj

        fj = (bi * fi - fp) / aj
        gj = gi * aj / j
        hj = hi + fj * gj
        sj = si + dj * hj
        return sj, rj, j, bj, cj, dj, fi, fj, gj, hj

    max_iter = 100

    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)
    if mask is not None:
        mask = tf.convert_to_tensor(mask, tf.bool)

    tol = tk.epsilon(x.dtype)
    n = tk.round(v)
    u = v - n

    i = tf.cast(1., x.dtype)
    a1 = 0.25 - tf.square(u)
    b1 = 2. * x + 2.

    c1 = 1. / b1
    d1 = a1 / b1
    r1 = d1

    f0 = tf.zeros_like(v * x)
    f1 = tf.ones_like(v * x)
    g1 = tf.ones_like(v * x)
    h1 = f1 * g1
    s1 = 1. + d1 * h1

    init = s1, r1, i, b1, c1, d1, f0, f1, g1, h1
    sn, rn, *_ = tf.while_loop(cond, body, init, maximum_iterations=max_iter)
    log_ku = 0.5 * tk.log(0.5 * tk.pi / x) - x - tk.log(sn)
    log_kup1 = log_ku + tk.log((0.5 + u + x - rn) / x)
    return log_ku, log_kup1
