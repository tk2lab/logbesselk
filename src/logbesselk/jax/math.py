import math

import jax
from jax.numpy import abs as fabs
from jax.numpy import cosh
from jax.numpy import exp
from jax.numpy import expm1
from jax.numpy import inf
from jax.numpy import isfinite as is_finite
from jax.numpy import log
from jax.numpy import log1p
from jax.numpy import logaddexp as log_add_exp
from jax.numpy import maximum
from jax.numpy import nan
from jax.numpy import round as fround
from jax.numpy import sign
from jax.numpy import sinc
from jax.numpy import sinh
from jax.numpy import sqrt
from jax.numpy import square
from jax.numpy import tanh
from jax.numpy import where

__all__ = [
    "fabs",
    "cosh",
    "exp",
    "expm1",
    "inf",
    "is_finite",
    "log",
    "log1p",
    "log_add_exp",
    "maximum",
    "nan",
    "fround",
    "sign",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "tanh",
    "where",
    "sinhc",
    "log_sinh",
    "log_cosh",
]


def func_with_vjp(gen):
    def wrap_fwd(*args, **kwargs):
        return fwd(wrap_func, *args, **kwargs)

    def wrap_bwd(res, upstream):
        return bwd(wrap_func, res, upstream)

    func, fwd, bwd = gen()
    # func = functools.wraps(gen)(func)
    wrap_func = jax.custom_vjp(func)
    wrap_func.defvjp(wrap_fwd, wrap_bwd)
    return wrap_func


def sinhc(x):
    return where(x == 0, 1, sinh(x) / x)


@func_with_vjp
def log_sinh():
    def func(x):
        return where(x < 20, log(sinh(x)), x - math.log(2))

    def fwd(wrap_func, x):
        return func(x), x

    def bwd(wrap_func, x, upstream):
        dx = 1 / tanh(x)
        return (upstream * dx,)

    return func, fwd, bwd


@func_with_vjp
def log_cosh():
    def func(x):
        return x + log1p(expm1(-2 * x) / 2)

    def fwd(wrap_func, x):
        return func(x), x

    def bwd(wrap_func, x, upstream):
        dx = tanh(x)
        return (upstream * dx,)

    return func, fwd, bwd
