import tensorflow as tf
import numpy as np

from . import math as tk
from .utils import log_bessel_recurrence


def log_K(v, x):
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


def _log_ku_temme(u, x):

    def cond(ki, li, i, ci, pi, qi, fi):
        return tf.reduce_any(
            (0. < x) & (x <= 2.) & (tk.abs(ci * fi) > tol * tk.abs(ki))
        )

    def body(ki, li, i, ci, pi, qi, fi):
        j = i + 1.
        cj = ci * tk.square(xh) / j
        pj = pi / (j - u)
        qj = qi / (j + u)
        fj = (j * fi + pi + qi) / (tk.square(j) - tk.square(u))
        kj = ki + cj * fj
        lj = li + cj * (pj - j * fj)
        return kj, lj, j, cj, pj, qj, fj

    dtype = (u * x).dtype
    shape = tf.shape(u * x)
    tol = tk.epsilon(dtype)

    u = tk.abs(u)
    gmuinv = tk.exp(-tk.log_gamma(1. - u))
    gpuinv = tk.exp(-tk.log_gamma(1. + u))
    gm = tf.where(u > tol, 0.5 * (gpuinv - gmuinv) / u, np.euler_gamma)
    gp = 0.5 * (gpuinv + gmuinv)
    xh = 0.5 * x
    lxh = tk.log(xh)
    mu = u * lxh

    i = tf.cast(0., dtype)
    c0 = tf.ones(shape, dtype)
    p0 = 0.5 * tk.exp(-mu) / gpuinv
    q0 = 0.5 * tk.exp( mu) / gmuinv
    f0 = -(gp * lxh * tk.sinhc(mu) + gm * tk.cosh(mu)) / tk.sinc(u)
    k0 = c0 * f0
    l0 = c0 * (p0 - i * f0)
    init = k0, l0, i, c0, p0, q0, f0

    ku, kn, *_ = tf.while_loop(cond, body, init, maximum_iterations=1000)
    return tk.log(ku), tk.log(kn / xh)
