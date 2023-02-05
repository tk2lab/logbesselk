import jax.lax as lax

from .math import (
    log,
    log_add_exp,
    sign,
)
from .utils import (
    result_type,
)

__all__ = [
    "sign_deriv_bessel_k",
    "log_bessel_recurrence",
]


def sign_deriv_bessel_k(v, x, m=0, n=0):
    dtype = result_type(v, x)
    out = dtype(1)
    if n % 2 == 1:
        out *= -1
    if m % 2 == 0:
        return out
    else:
        return sign(v) * out


def log_bessel_recurrence(log_ku, log_kup1, u, n, x):
    def cond(args):
        ki, kj, ui, ni = args
        return ni > 0

    def body(args):
        ki, kj, ui, ni = args
        uj = ui + 1
        nj = ni - 1
        kk = log_add_exp(ki, kj + log(2 * uj / x))
        return kj, kk, uj, nj

    init = log_ku, log_kup1, u, n
    return lax.while_loop(cond, body, init)[:2]
