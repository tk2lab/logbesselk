import tensorflow as tf
import numpy as np

from . import math as tk
from .utils import wrap_K, wrap_log_K
from .utils import log_bessel_recurrence


def K(v, x, name=None):
    return wrap_K(_log_K, None, None, v, x, name or 'K_cf2')


def log_K(v, x, name=None):
    return wrap_log_K(_log_K, None, None, v, x, name or 'log_K_cf2')



def _log_K(v, x):
    """
    I.J. Thompson and A.R. Barnett,
    Modified Bessel function Iv(z) and Kv(z) and real order
    and complex argument, to selected accuracy,
    Computer Physics Communications, 47, 245-257 (1987).
    """
    n = tk.round(v)
    u = v - n
    log_ku0, log_ku1 = _log_ku_cf2(u, x)
    return log_bessel_recurrence(log_ku0, log_ku1, u, n, x)[0]


def _log_ku_cf2(u, x, mask=None):

    def cond(si, ri, i, ai, bi, gi, hi, qi, qj, ci, ti):
        return tf.reduce_any(mask & (tk.abs(hi * ti) > tol * tk.abs(si)))

    def body(si, ri, i, ai, bi, gi, hi, qm, qi, ci, ti):
        j = i + 1.

        aj = ai + 2. * j
        bj = bi + 2.

        gj = 1. / (bj - ai * gi)
        hj = hi * (bj * gj - 1.)

        qj = (bi * qi - qm) / ai
        cj = ci * ai / j
        tj = ti + cj * qj

        sj = si + hj * tj
        rj = ri + hj
        return sj, rj, j, aj, bj, gj, hj, qi, qj, cj, tj

    dtype = (u * x).dtype
    shape = tf.shape(u * x)

    tol = tk.epsilon(dtype)
    if mask is None:
        mask = tf.ones(shape, tf.bool)

    a0 = 0.25 - tk.square(u)
    b0 = 2. * x
    s0, r0 = 1., 0.

    i = tf.cast(1., dtype)
    a1 = a0 + 2.
    b1 = b0 + 2.
    g1 = 1. / b1
    h1 = 1. / b1
    q0 = tf.zeros(shape, dtype)
    q1 = tf.ones(shape, dtype)
    c1 = a0
    t1 = a0
    s1 = s0 + h1 * t1
    r1 = r0 + h1
    init = s1, r1, i, a1, b1, g1, h1, q0, q1, c1, t1

    sn, rn, *_ = tf.while_loop(cond, body, init, maximum_iterations=1000)
    log_ku = 0.5 * tk.log(np.pi / b0) - x - tk.log(sn)
    log_kup1 = log_ku + tk.log((0.5 + u + x - a0 * rn) / x)
    return log_ku, log_kup1
