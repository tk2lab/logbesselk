import tensorflow as tf
import numpy as np

from . import math as tk


def log_K_temme(v, x):
    n = tk.round(v)
    u = v - n
    ku0_small, ku1_small = _ku_temme_original(u, x)
    ku0_large, ku1_large = _ku_temme_cambell(u, x)
    ku0 = tf.where(x <= 2, ku0_small, ku0_large)
    ku1 = tf.where(x <= 2, ku1_small, ku1_large)
    return _bessel_recurrence(ku0, ku1, u, x, n)[0]


def K_ratio_temme(v, x):
    n = tk.round(v)
    u = v - n
    ku0_small, ku1_small = _ku_temme_original(u, x)
    ku0_large, ku1_large = _ku_temme_cambell(u, x)
    ku0 = tf.where(x <= 2, ku0_small, ku0_large)
    ku1 = tf.where(x <= 2, ku1_small, ku1_large)
    ku0, ku1, _ = _bessel_recurrence(ku0, ku1, u, x, n)
    return tk.exp(ku1 - ku0)


def log_K_temme_original(v, x):
    """The modified Bessel function of the second kind in log-space.
    References:
    [1] N. Temme. On the numerical evaluation of the modified Bessel function
        of the third kind. Journal of Coumputational Physics 19, 1975.
    [2] Numericcal Recipes in C. The Art of Scientific Coumputing, 2nd Edition,
        1992.
    """
    n = tk.round(v)
    u = v - n
    ku0, ku1 = _ku_temme_original(u, x)
    return _bessel_recurrence(ku0, ku1, u, x, n)[0]


def log_K_temme_cambell(v, x):
    """The modified Bessel function of the second kind in log-space.
    Refeerences
    [1] N. Temme, On the numerical evaluation of the modified Bessel function
        of the third kind. Journal of Computational Physics 19, 1975.
    [2] J. Cambell. On Temme's algorithm for the modified Bessel function
        of the third kind. https://dl.acm.org/doi/pdf/10.1145/355921.35928
    [3] Numerical recipes in C. The Art of Scientfic Computing,
        2nd Edition, 1992.
    """
    n = tk.round(v)
    u = v - n
    ku0, ku1 = _ku_temme_cambell(u, x)
    return _bessel_recurrence(ku0, ku1, u, x, n)[0]


def _bessel_recurrence(ku0, ku1, u, x, n):

    def cond(k0, k1, ui, ni):
        return tf.reduce_any(ni != 0.)

    def body(k0, k1, ui, ni):
        kp = tk.log_add_exp(k0, k1 + tk.log(2. * (ui + 1.) / x))
        km = tk.log_sub_exp(k1, k0 + tk.log(2. * ui / x))
        k0 = tf.where(tf.equal(ni, 0.), k0, tf.where(ni > 0., k1, km))
        k1 = tf.where(tf.equal(ni, 0.), k1, tf.where(ni > 0., kp, k0))
        ui += tk.sign(ni)
        ni -= tk.sign(ni)
        return k0, k1, ui, ni

    init = ku0, ku1, u, n
    return tf.while_loop(cond, body, init)


def _ku_temme_original(u, x):

    def calc_coef(u):
        factors = [[
            -1.142022680371168e0,
            6.5165112670737e-3,
            3.087090173086e-4,
            -3.4706269649e-6,
            6.9437664e-9,
            3.67795e-11,
            -1.356e-13,
        ], [
            1.843740587300905e0,
            -7.68528408447867e-2,
            1.2719271366546e-3,
            -4.9717367042e-6,
            -3.31261198e-8,
            2.423096e-10,
            -1.702e-13,
            -1.49e-15,
        ]]
        w = 16. * tk.square(u) - 2.
        coef = []
        for fk in factors:
            prev, curr = 0., 0.
            for fi in fk[:0:-1]:
                prev, curr = curr, (w * curr + fi) - prev
            coef.append((w * curr + fk[0]) / 2. - prev)
        return coef

    def cond(k0, k1, f, p, q, c, i):
        return tf.reduce_any(
            (0. < x) & (x <= 2.) & (tk.abs(c * f) > tol * tk.abs(k0))
        )

    def body(k0, k1, f, p, q, c, i):
        f = (i * f + p + q) / (tk.square(i) - tk.square(u))
        p /= (i - u)
        q /= (i + u)
        c *= tk.square(x) / (4. * i)
        k0 = k0 + c * f
        k1 = k1 + c * (p - i * f)
        i += 1.
        return k0, k1, f, p, q, c, i

    tol = tk.epsilon((u * x).dtype)
    coef0, coef1 = calc_coef(u)
    lx = tk.log(x / 2.)
    mu = -u * lx

    f = (tk.cosh(mu) * coef0 - lx * tk.sinhc(mu) * coef1) / tk.sinc(u)
    p = tk.exp( mu) / (2. * (coef1 - u * coef0))
    q = tk.exp(-mu) / (2. * (coef1 + u * coef0))
    c = tf.ones_like(u * x)
    i = tf.cast(1., (u * x).dtype)
    k0 = f
    k1 = p
    init = f, p, f, p, q, c, i

    k0, k1, *_ = tf.while_loop(cond, body, init, maximum_iterations=1000)
    return tf.math.log(k0), tf.math.log(2. * k1 / x)


def _ku_temme_cambell(u, x):

    def cond(hr, hs, k0, k1, dr, cd, i, c, q):
        return tf.reduce_any((x > 2.) & (tk.abs(cd * q) > tol * tk.abs(hs)))

    def body(hr, hs, k0, k1, dr, cd, i, c, q):
        pn = tk.square(u) - tk.square(i + 0.5) 
        pd = 2. * (x + i)
        k0, k1 = k1, (k0 - pd * k1) / pn
        dr = 1. / ((pd + 2.) + pn * dr)
        cd *= (pd + 2.) * dr - 1.
        i += 1.
        c *= -pn / i
        q += c * k1
        hr += cd
        hs += cd * q
        return hr, hs, k0, k1, dr, cd, i, c, q

    tol = tk.epsilon((u * x).dtype)
    pn = tk.square(u) - 0.25
    pd = 2. * x + 2.

    hr = 1 / pd
    hs = 1. - pn / pd
    k0 = tf.zeros_like(u * x)
    k1 = tf.ones_like(u * x)
    dr = 1. / pd
    cd = 1. / pd
    i = tf.cast(1., (u * x).dtype)
    c = -pn
    q = -pn
    init = hr, hs, k0, k1, dr, cd, i, c, q

    hr, hs, *_ = tf.while_loop(cond, body, init, maximum_iterations=1000)
    log_k0 = tk.log(np.pi / (2. * x)) / 2. - tk.log(hs) - x
    log_k1 = log_k0 + tk.log((1. + 2. * (u + x + pn * hr)) / (2. * x))
    return log_k0, log_k1
