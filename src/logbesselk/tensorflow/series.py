import tensorflow as tf

from .math import cosh
from .math import exp
from .math import fabs
from .math import log
from .math import sinc
from .math import sinhc
from .math import square
from .misc import log_bessel_recurrence
from .utils import epsilon
from .utils import result_shape
from .utils import result_type
from .wrap import wrap_log_bessel_k

__all__ = [
    "log_bessel_k",
]


@wrap_log_bessel_k
def log_bessel_k(v, x):
    """
    N.M. Temme.
    On the numerical evaluation of the modified Bessel function
    of the third kind.
    Journal of Coumputational Physics, 19, 324-337 (1975).
    """
    n = tf.math.round(v)
    u = v - n
    log_ku, log_kup1 = log_bessel_ku(u, x)
    return log_bessel_recurrence(log_ku, log_kup1, u, n, x)[0]


def log_bessel_ku(u, x):
    def cond(si, li, i, di, pi, qi, hi):
        nonzero_update = fabs(di * hi) >= eps * fabs(si)
        return tf.math.reduce_any(nonzero_update)

    def body(si, li, i, di, pi, qi, hi):
        j = i + 1
        dj = di * square(x / 2) / j
        pj = pi / (j - u)
        qj = qi / (j + u)
        hj = (j * hi + pi + qi) / (square(j) - square(u))
        sj = si + dj * hj
        lj = li + dj * (pj - j * hj)
        return sj, lj, j, dj, pj, qj, hj

    tf.print(u, x)
    max_iter = 100

    shape = result_shape(u, x)
    dtype = result_type(u, x)
    eps = epsilon(dtype)

    gp, gm = calc_gamma(u)
    lxh = log(x / 2)
    mu = u * lxh

    i = tf.constant(0, dtype)
    d0 = tf.ones(shape, dtype)
    p0 = (1 / 2) * exp(-mu) / (gp - u * gm)
    q0 = (1 / 2) * exp(mu) / (gp + u * gm)
    h0 = (gm * cosh(mu) - gp * lxh * sinhc(mu)) / sinc(u)
    s0 = d0 * h0
    l0 = d0 * (p0 - i * h0)

    init = s0, l0, i, d0, p0, q0, h0
    ku, kn, *_ = tf.while_loop(cond, body, init, maximum_iterations=max_iter)
    return log(ku), log(kn) - lxh


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
    w = 16 * square(u) - 2
    coef = [None, None]
    for s in range(2):
        prev, curr = 0, 0
        for fac in reversed(factor[s + 2 :: 2]):
            prev, curr = curr, w * curr + fac - prev
        coef[s] = (1 / 2) * (w * curr + factor[s]) - prev
    return coef
