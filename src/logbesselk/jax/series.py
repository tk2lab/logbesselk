import jax.lax as lax
import jax.numpy as jnp

from .math import sinhc
from .wrap import log_bessel_recurrence
from .wrap import wrap_simple

__all__ = [
    "log_bessel_k",
]


@wrap_simple
def log_bessel_k(v, x):
    """
    N. M. Temme.
    On the numerical evaluation of the modified Bessel function
    of the third kind.
    Journal of Coumputational Physics, 19, 324-337 (1975).
    """
    n = jnp.round(v)
    u = v - n
    log_ku, log_kup1 = log_bessel_ku(u, x)
    return log_bessel_recurrence(log_ku, log_kup1, u, n, x)[0]


def log_bessel_ku(u, x):
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
        w = 16 * jnp.square(u) - 2
        coef = [None, None]
        for s in range(2):
            prev, curr = 0, 0
            for fac in reversed(factor[s + 2 :: 2]):
                prev, curr = curr, w * curr + fac - prev
            coef[s] = (1 / 2) * (w * curr + factor[s]) - prev
        return coef

    def cond(args):
        ki, li, i, ci, pi, qi, fi = args
        return (i < max_iter) & (jnp.abs(ci * fi) > eps * jnp.abs(ki))

    def body(args):
        ki, li, i, ci, pi, qi, fi = args
        j = i + 1
        cj = ci * jnp.square(x / 2) / j
        pj = pi / (j - u)
        qj = qi / (j + u)
        fj = (j * fi + pi + qi) / (jnp.square(j) - jnp.square(u))
        kj = ki + cj * fj
        lj = li + cj * (pj - j * fj)
        return kj, lj, j, cj, pj, qj, fj

    max_iter = 100

    dtype = jnp.asarray(u * x).dtype
    eps = jnp.finfo(dtype).eps

    gp, gm = calc_gamma(u)
    lxh = jnp.log(x / 2)
    mu = u * lxh

    i = 0
    c0 = jnp.ones((), dtype)
    p0 = (1 / 2) * jnp.exp(-mu) / (gp - u * gm)
    q0 = (1 / 2) * jnp.exp(+mu) / (gp + u * gm)
    f0 = (gm * jnp.cosh(mu) - gp * lxh * sinhc(mu)) / jnp.sinc(u)
    k0 = c0 * f0
    l0 = c0 * (p0 - i * f0)
    init = k0, l0, i, c0, p0, q0, f0
    ku, kn, counter, *_ = lax.while_loop(cond, body, init)
    return jnp.log(ku), jnp.log(kn) - lxh
