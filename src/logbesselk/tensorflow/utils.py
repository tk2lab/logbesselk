import numpy as np
import tensorflow as tf

from .math import (
    exp,
    fabs,
    log,
)

__all__ = [
    "result_shape",
    "result_type",
    "epsilon",
    "grad",
    "extend",
    "find_zero",
    "log_integrate",
]


def result_shape(*args):
    x = args[0]
    for a in args[1:]:
        x *= a
    return tf.shape(x)


def as_numpy_dtype(dtype):
    dtype = tf.as_dtype(dtype)
    if hasattr(dtype, "as_numpy_dtype"):
        return dtype.as_numpy_dtype
    return dtype


def result_type(*args, dtype_hint=None):
    args = tf.nest.flatten(args)
    dtype = None
    for i, a in enumerate(args):
        if hasattr(a, "dtype") and a.dtype:
            dt = as_numpy_dtype(a.dtype)
            if dtype is None:
                dtype = dt
            elif dtype != dt:
                dtype = np.ones([2], dtype) * np.ones([2], dt).dtype
    return dtype_hint if dtype is None else tf.as_dtype(dtype)


def epsilon(dtype):
    return np.finfo(dtype.as_numpy_dtype).eps


def grad(func, i=0):
    def deriv(*args):
        with tf.GradientTape() as g:
            g.watch(args)
            f = func(*args)
        return g.gradient(f, args[i])

    return deriv


def extend(func, x0, dx):
    def cond(x, d, f1):
        return tf.reduce_any(~tf.equal(dx, 0) & (f1 > 0))

    def body(x, d, f1):
        f1 = func(x + d)
        x = tf.where(f1 > 0, x + d, x)
        d = tf.where(f1 > 0, 2 * d, d)
        return x, d, f1

    dummy = tf.ones_like(x0)
    x, d, _ = tf.while_loop(cond, body, (x0, dx, dummy))
    return x, d


def find_zero(func, x0, dx, tol, max_iter):
    def cond(x0, x1):
        f0 = func(x0)
        return tf.reduce_any(~tf.equal(x0, x1) & (fabs(f0) > tol))

    def body(x0, x1):
        f0 = func(x0)
        f1 = func(x1)

        x_shrink = x0 + 0.5 * (x1 - x0)
        f_shrink = func(x_shrink)
        cond = f_shrink * f0 < 0
        x0, x1 = x_shrink, tf.where(cond, x0, x1)
        f0, f1 = f_shrink, tf.where(cond, f0, f1)

        diff = -f0 / deriv(x0)
        ddx = diff / (x1 - x0)
        dx_in_range = (0 < ddx) & (ddx < 1)
        x_newton = tf.where(dx_in_range, x0 + diff, x0)
        f_newton = func(x_newton)
        x0, x1 = x_newton, tf.where(f_newton * f0 < 0, x0, x1)
        x1 = tf.where(f_newton * f0 < 0, x0, x1)
        x0 = x_newton
        return x0, x1

    deriv = grad(func)
    init = x0, x0 + dx
    return tf.while_loop(cond, body, init, maximum_iterations=max_iter)[0]


def log_integrate(func, t0, t1, bins):
    def cond(fmax, fsum, i):
        return i < bins

    def body(fmax, fsum, i):
        a = tf.cast(2 * i + 1, dtype) / tf.cast(2 * bins, dtype)
        t = (1 - a) * t0 + a * t1
        ft = func(t)
        diff = ft - fmax
        keep_fmax = ft < fmax
        fsum = tf.where(
            keep_fmax,
            fsum + exp(diff),
            fsum * exp(-diff) + 1,
        )
        fmax = tf.where(keep_fmax, fmax, ft)
        return fmax, fsum, i + 1

    shape = result_shape(t0, t1)
    dtype = result_type(t0, t1)
    zero = tf.zeros(shape, dtype)
    izero = tf.constant(0, tf.int32)
    bins = tf.constant(bins, tf.int32)
    fmax, fsum, _ = tf.while_loop(cond, body, (zero, zero, izero))
    h = fabs(t1 - t0) / tf.cast(bins, dtype)
    return fmax + log(fsum) + log(h)
