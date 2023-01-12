import math

import jax
import jax.numpy as jnp

__all__ = [
    "sinhc",
    "log_sink",
    "log_cosh",
]


def sinhc(x):
    return jnp.where(x == 0, 1, jnp.sinh(x) / x)


@jax.custom_vjp
def log_sinh(x):
    return jnp.where(x < 20, jnp.log(jnp.sinh(x)), x - math.log(2))


def log_sinh_fwd(x):
    return log_sinh(x), x


def log_sinh_bwd(x, g):
    return (g / jnp.tanh(x),)


log_sinh.defvjp(log_sinh_fwd, log_sinh_bwd)


@jax.custom_vjp
def log_cosh(x):
    return x + jnp.log1p(jnp.expm1(-2 * x) / 2)


def log_cosh_fwd(x):
    return log_cosh(x), x


def log_cosh_bwd(x, g):
    return (g * jnp.tanh(x),)


log_cosh.defvjp(log_cosh_fwd, log_cosh_bwd)
