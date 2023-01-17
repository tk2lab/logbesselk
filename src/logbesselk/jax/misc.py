import jax.numpy as jnp

from .wrap import sign_bessel_k

__all__ = [
    "wrap_slog_bessel_k",
    "wrap_bessel_ke",
    "wrap_bessel_k_ratio",
]


def wrap_slog_bessel_k(log_bessel_k, v, x, m=0, n=0):
    sign = sign_bessel_k(v, x, m, n)
    logk = log_bessel_k(v, x, m, n)
    return sign, logk


def wrap_bessel_ke(log_bessel_k, v, x):
    sign = sign_bessel_k(v, x)
    logk = log_bessel_k(v, x)
    return sign * jnp.exp(logk + x)


def wrap_bessel_k_ratio(log_bessel_k, v, x, d=1):
    signd = sign_bessel_k(v + d, x)
    logkd = log_bessel_k(v + d, x)
    sign = sign_bessel_k(v, x)
    logk = log_bessel_k(v, x)
    return signd * sign * jnp.exp(logkd - logk)
