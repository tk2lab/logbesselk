import tensorflow as tf
import numpy as np

from . import math as tk
from .utils import bessel_recurrence


def log_K_temme(v, x):
    """
    N.M. Temme.
    On the numerical evaluation of the modified Bessel function
    of the third kind.
    Journal of Coumputational Physics, 19, 324-337 (1975).
    """
    n = tk.round(v)
    u = v - n
    ku0, ku1 = _ku_temme(u, x)
    return bessel_recurrence(ku0, ku1, u, x, n)[0]


def _ku_temme(u, x):

    def cond(ki, li, i, pi, qi, ci, fi):
        return tf.reduce_any(
            (0. < x) & (x <= 2.) & (tk.abs(ci * fi) > tol * tk.abs(ki))
        )

    def body(ki, li, i, pi, qi, ci, fi):
        j = i + 1.
        pj = pi / (j - u)
        qj = qi / (j + u)
        cj = ci * tk.square(xh) / i
        fj = (j * fi + pi + qi) / (tk.square(j) - tk.square(u))
        kj = ki + cj * fj
        lj = li + cj * (pj - j * fj)
        return kj, lj, j, pj, qj, cj, fj

    dtype = (u * x).dtype
    shape = tf.shape(u * x)
    tol = tk.epsilon(dtype)

    lgm = tk.log_gamma(1. - u)
    lgp = tk.log_gamma(1. + u)
    g1 = 0.5 * tk.sinc(u) * (tk.exp(lgp) - tk.exp(lgm))
    g2 = 0.5 * tk.sinc(u) * (tk.exp(lgp) + tk.exp(lgm) + 2.)
    xh = 0.5 * x
    lxh = tk.log(xh)

    i = tf.cast(1., dtype)
    p0 = 0.5 * tk.exp(lgm - u * lxh)
    q0 = 0.5 * tk.exp(lgp + u * lxh)
    c0 = tf.ones(shape, dtype)
    f0 = (tk.cosh(u * lxh) * g1 + lxh * tk.sinh(u * lxh) * g2) / tk.sinc(u)
    k0 = c0 * f0
    l0 = c0 * (p0 - i * f0)
    init = k0, l0, i, p0, q0, c0, f0

    ku, kn, *_ = tf.while_loop(cond, body, init, maximum_iterations=1000)
    return ku, kn / xh
