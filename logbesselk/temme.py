import tensorflow as tf
import numpy as np

from . import math as tk
from .utils import wrap_K, wrap_log_K
from .utils import log_bessel_recurrence


def K(v, x, name=None):
    return wrap_K(_log_K, None, None, v, x, name or 'bessel_K_tem')


def log_K(v, x, name=None):
    return wrap_log_K(_log_K, None, None, v, x, name or 'log_K_tem')



def _log_K(v, x):
    """
    N.M. Temme.
    On the numerical evaluation of the modified Bessel function
    of the third kind.
    Journal of Coumputational Physics, 19, 324-337 (1975).
    """
    n = tk.round(v)
    u = v - n
    log_ku, log_kup1 = _log_ku_temme(u, x)
    return log_bessel_recurrence(log_ku, log_kup1, u, n, x)[0]


def _log_ku_temme(u, x, mask=None):

    def calc_gamma(u):
        factor = [
            +1.8437405873009050, -1.1420226803711680,
            -0.0768528408447867, +0.0065165112670737,
            +0.0012719271366546, +0.0003087090173086,
            -0.0000049717367042, -0.0000034706269649,
            -0.0000000331261198, +0.0000000069437664,
            +0.0000000002423096, +0.0000000000367795,
            -0.0000000000001702, -0.0000000000001356,
            -0.00000000000000149,
        ]
        w = 16. * tk.square(u) - 2.
        coef = [None, None]
        for s in range(2):
            prev, curr = 0., 0.
            for fac in reversed(factor[s + 2::2]):
                prev, curr = curr, w * curr + fac - prev
            coef[s] = 0.5 * (w * curr + factor[s]) - prev
        return coef

    def cond(ki, li, i, ci, pi, qi, fi):
        return tf.reduce_any(mask & (tk.abs(ci * fi) > tol * tk.abs(ki)))

    def body(ki, li, i, ci, pi, qi, fi):
        j = i + 1.
        cj = ci * tk.square(0.5 * x) / j
        pj = pi / (j - u)
        qj = qi / (j + u)
        fj = (j * fi + pi + qi) / (tk.square(j) - tk.square(u))
        kj = ki + cj * fj
        lj = li + cj * (pj - j * fj)
        return kj, lj, j, cj, pj, qj, fj

    dtype = (u * x).dtype
    shape = tf.shape(u * x)

    tol = tk.epsilon(dtype)
    if mask is None:
        mask = tf.ones(shape, tf.bool)

    gp, gm = calc_gamma(u)
    lxh = tk.log(0.5 * x)
    mu = u * lxh

    i = tf.cast(0., dtype)
    c0 = tf.ones(shape, dtype)
    p0 = 0.5 * tk.exp(-mu) / (gp - u * gm)
    q0 = 0.5 * tk.exp( mu) / (gp + u * gm)
    f0 = (gm * tk.cosh(mu) - gp * lxh * tk.sinhc(mu)) / tk.sinc(u)
    k0 = c0 * f0
    l0 = c0 * (p0 - i * f0)
    init = k0, l0, i, c0, p0, q0, f0

    ku, kn, *_ = tf.while_loop(cond, body, init, maximum_iterations=1000)
    return tk.log(ku), tk.log(kn) - lxh
