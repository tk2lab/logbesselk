import tensorflow as tf

from . import math as tk
from .utils import log_bessel_recurrence
from .utils import wrap_log_k

__all__ = [
    "log_bessel_k",
]


@wrap_log_k
def log_bessel_k(v, x):
    v = tk.abs(v)
    n = tk.round(v)
    u = v - n
    log_ku, log_kup1 = _log_bessel_ku(u, x)
    return log_bessel_recurrence(log_ku, log_kup1, u, n, x)[0]


def _log_bessel_ku(u, x, mask=None, max_iter=100, return_counter=False):
    """
    N.M. Temme.
    On the numerical evaluation of the modified Bessel function
    of the third kind.
    Journal of Coumputational Physics, 19, 324-337 (1975).
    """

    def calc_gamma(u):
        factor = [
            +1.8437405873009050,
            -1.1420226803711680,
            -0.0768528408447867,
            +0.0065165112670737,
            +0.0012719271366546,
            +0.0003087090173086,
            -0.0000049717367042,
            -0.0000034706269649,
            -0.0000000331261198,
            +0.0000000069437664,
            +0.0000000002423096,
            +0.0000000000367795,
            -0.0000000000001702,
            -0.0000000000001356,
            -0.00000000000000149,
        ]
        w = 16 * tk.square(u) - 2
        coef = [None, None]
        for s in range(2):
            prev, curr = 0, 0
            for fac in reversed(factor[s + 2 :: 2]):
                prev, curr = curr, w * curr + fac - prev
            coef[s] = (1 / 2) * (w * curr + factor[s]) - prev
        return coef

    def cond(ki, li, i, ci, pi, qi, fi):
        nonzero_update = tk.abs(ci * fi) > tol * tk.abs(ki)
        if mask is not None:
            nonzero_update &= mask
        return tf.reduce_any(nonzero_update)

    def body(ki, li, i, ci, pi, qi, fi):
        j = i + 1
        cj = ci * tk.square(x / 2) / j
        pj = pi / (j - u)
        qj = qi / (j + u)
        fj = (j * fi + pi + qi) / (tk.square(j) - tk.square(u))
        kj = ki + cj * fj
        lj = li + cj * (pj - j * fj)
        return kj, lj, j, cj, pj, qj, fj

    x = tf.convert_to_tensor(x)
    u = tf.convert_to_tensor(u, x.dtype)
    if mask is not None:
        mask = tf.convert_to_tensor(mask, tf.bool)

    tol = tk.epsilon(x.dtype)
    gp, gm = calc_gamma(u)
    lxh = tk.log(x / 2)
    mu = u * lxh

    i = tf.cast(0, x.dtype)
    c0 = tf.ones_like(u * x)
    p0 = (1 / 2) * tk.exp(-mu) / (gp - u * gm)
    q0 = (1 / 2) * tk.exp(mu) / (gp + u * gm)
    f0 = (gm * tk.cosh(mu) - gp * lxh * tk.sinhc(mu)) / tk.sinc(u)
    k0 = c0 * f0
    l0 = c0 * (p0 - i * f0)
    init = k0, l0, i, c0, p0, q0, f0

    ku, kn, counter, *_ = tf.while_loop(
        cond,
        body,
        init,
        maximum_iterations=max_iter,
    )
    log_ku, log_kup1 = tk.log(ku), tk.log(kn) - lxh

    if return_counter:
        return log_ku, log_kup1, counter

    return log_ku, log_kup1
