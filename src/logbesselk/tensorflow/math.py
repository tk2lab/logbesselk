import functools
import math

import tensorflow as tf
from tensorflow import (
    where,
)
from tensorflow.math import abs as fabs
from tensorflow.math import (
    cosh,
    exp,
    expm1,
    is_finite,
    log,
    log1p,
    maximum,
)
from tensorflow.math import round as fround
from tensorflow.math import (
    sign,
    sin,
    sinh,
    sqrt,
    square,
    tanh,
)

__all__ = [
    "fabs",
    "cosh",
    "exp",
    "expm1",
    "is_finite",
    "log",
    "log1p",
    "maximum",
    "fround",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tanh",
    "sinc",
    "sinhc",
    "log_sinh",
    "log_cosh",
    "where",
]


def sinc(x):
    pix = math.pi * x
    return where(
        tf.equal(x, 0),
        tf.constant(1, x.dtype),
        sin(pix) / pix,
    )


def sinhc(x):
    return where(
        tf.equal(x, 0),
        tf.constant(1, x.dtype),
        sinh(x) / x,
    )


def log_add_exp(x, y, sign=None):
    larger = maximum(x, y)
    if sign is None:
        sign = 1
    return larger + log(exp(x - larger) + sign * exp(y - larger))


def func_with_vjp(gen):
    def wrap_func(*args, **kwargs):
        out, res = fwd(custom_func, *args, **kwargs)

        def grad(upstream):
            return bwd(custom_func, res, upstream)

        return out, grad

    func, fwd, bwd = gen()
    func = functools.wraps(gen)(func)
    custom_func = tf.custom_gradient(wrap_func)
    return custom_func


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
