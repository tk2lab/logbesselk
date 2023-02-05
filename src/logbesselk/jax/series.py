import jax.lax as lax

from .math import (
    cosh,
    exp,
    fabs,
    fround,
    log,
    sinc,
    sinhc,
    square,
)
from .misc import (
    log_bessel_recurrence,
)
from .utils import (
    epsilon,
    result_type,
)
from .wrap import (
    wrap_log_bessel_k,
)

__all__ = [
    "log_bessel_k",
]


@wrap_log_bessel_k
def log_bessel_k(v, x):
    """
    N. M. Temme.
    On the numerical evaluation of the modified Bessel function
    of the third kind.
    Journal of Coumputational Physics, 19, 324-337 (1975).
    """
    n = fround(v)
    u = v - n
    log_ku, log_kup1 = log_bessel_ku(u, x)
    return log_bessel_recurrence(log_ku, log_kup1, u, n, x)[0]


def log_bessel_ku(u, x):
    def cond(args):
        ku, kn, i, p, q, r, s = args
        update = fabs(r * s) > eps * fabs(ku)
        return (i < 10) | (update & (i < max_iter))

    def body(args):
        ku, kn, i, p, q, r, s = args
        i += 1
        p, q, r, s = (
            p / (i - u),
            q / (i + u),
            r * square(x / 2) / i,
            (p + q + i * s) / (square(i) - square(u)),
        )
        ku += r * s
        kn += r * (p - i * s)
        return ku, kn, i, p, q, r, s

    max_iter = 100

    dtype = result_type(u, x)
    eps = epsilon(dtype)

    gp, gm = calc_gamma(u)
    lxh = log(x / 2)
    mu = u * lxh

    i = 0
    p = (1 / 2) * exp(-mu) / (gp - u * gm)
    q = (1 / 2) * exp(mu) / (gp + u * gm)
    r = dtype(1)
    s = (gm * cosh(mu) - gp * lxh * sinhc(mu)) / sinc(u)

    ku = r * s
    kn = r * (p - i * s)

    init = ku, kn, i, p, q, r, s
    ku, kn, *_ = lax.while_loop(cond, body, init)
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
