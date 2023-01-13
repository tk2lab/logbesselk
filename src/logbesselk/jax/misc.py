import jax.numpy as jnp

from .integral import log_bessel_k
from .wrap import sign_bessel_k

__all__ = [
    "slog_bessel_k",
    "bessel_ke",
    "bessel_k_ratio",
]


def slog_bessel_k(v, x, m=0, n=0, log_bessel_k=log_bessel_k):
    sign = sign_bessel_k(v, x, m, n)
    logk = log_bessel_k(v, x, m, n)
    return sign, logk


def bessel_ke(v, x, m=0, n=0, log_bessel_k=log_bessel_k):
    sign = sign_bessel_k(v, x, m, n)
    logk = log_bessel_k(v, x, m, n)
    return sign * jnp.exp(logk + x)


def bessel_k_ratio(v, x, d=1, m=0, n=0, log_bessel_k=log_bessel_k):
    signd = sign_bessel_k(v + d, x, m, n)
    logkd = log_bessel_k(v + d, x, m, n)
    sign = sign_bessel_k(v, x, m, n)
    logk = log_bessel_k(v, x, m, n)
    return signd * sign * jnp.exp(logkd - logk)
