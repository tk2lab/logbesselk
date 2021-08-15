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

    def cond(si, ri, i, ai, bi, gi, hi, qi, qj, ci, ti):
        nonzero_update = tk.abs(hi * ti) > tol * tk.abs(si)
        if mask is not None:
            nonzero_update &= mask
        return tf.reduce_any(nonzero_update)

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

    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)
    if mask is not None:
        mask = tf.convert_to_tensor(mask, tf.bool)

    tol = tk.epsilon(x.dtype)
    n = tk.round(v)
    u = v - n
    a0 = 0.25 - tk.square(u)
    b0 = 2. * x
    s0, r0 = 1., 0.

    i = tf.cast(1., x.dtype)
    a1 = a0 + 2.
    b1 = b0 + 2.
    g1 = 1. / b1
    h1 = 1. / b1
    q0 = tf.zeros_like(v * x, x.dtype)
    q1 = tf.ones_like(v * x, x.dtype)
    c1 = a0
    t1 = a0
    s1 = s0 + h1 * t1
    r1 = r0 + h1
    init = s1, r1, i, a1, b1, g1, h1, q0, q1, c1, t1

    sn, rn, *_ = tf.while_loop(cond, body, init, maximum_iterations=1000)
    log_ku = 0.5 * tk.log(tk.pi / b0) - x - tk.log(sn)
    log_kup1 = log_ku + tk.log((0.5 + u + x - a0 * rn) / x)
    return log_ku, log_kup1
