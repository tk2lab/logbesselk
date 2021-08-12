import tensorflow as tf
import numpy as np

from . import math as tk
from .utils import bessel_recurrence


def log_K_cf2(v, x, name=None):
    """
    I.J. Thompson and A.R. Barnett,
    Modified Bessel function Iv(z) and Kv(z) and real order
    and complex argument, to selected accuracy,
    Computer Physics Communications, 47, 245-257 (1987).
    """
    n = tk.round(v)
    u = v - n
    ku0, ku1 = _ku_cf2(u, x)
    return bessel_recurrence(ku0, ku1, u, x, n)[0]


def _ku_cf2(u, x):

    def a(k):
        k = tf.convert_to_tensor(k, dtype)
        return tk.square(k + 0.5) - tk.square(u)

    def b(k):
        return 2. * (x + k)

    def cond(si, ri, i, gi, hi, qi, qj, ci, ti):
        return tf.reduce_any((x > 2.) & (tk.abs(hi * ti) > tol * tk.abs(si)))

    def body(si, ri, i, gi, hi, qi, qj, ci, ti):
        j = i + 1.

        gj = 1. / (b(j) - a(i) * gi)
        hj = hi * (b(j) * gj - 1.)

        qk = (b(i) * qj - qi) / a(i)
        cj = ci * a(i) / j
        tj = ti + qk * cj

        sj = si + hj * tj
        rj = ri + hj
        return sj, rj, j, gj, hj, qj, qk, cj, tj

    dtype = (u * x).dtype
    shape = tf.shape(u * x)
    tol = tk.epsilon(dtype)

    s0, r0 = 1., 0.
    i = tf.cast(1., dtype)
    g1 = 1. / b(1.)
    h1 = 1. / b(1.)
    q1 = tf.zeros(shape, dtype)
    q2 = tf.ones(shape, dtype)
    c1 = a(0.)
    t1 = a(0.)
    s1 = s0 + h1 * t1
    r1 = r0 + h1
    init = s1, r1, i, g1, h1, q1, q2, c1, t1

    sn, rn, *_ = tf.while_loop(cond, body, init, maximum_iterations=1000)
    ku = tk.sqrt(np.pi / b(0.)) * tk.exp(-x) / sn
    kup1 = ku * (0.5 + u + x - a(0.) * rn) / x
    return ku, kup1
