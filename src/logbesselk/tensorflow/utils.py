import numpy as np
import tensorflow as tf

__all__ = [
    "result_shape",
    "result_type",
    "epsilon",
    "get_deriv_func",
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


def get_deriv_func(func, i=0):
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

    init = x0, dx, tf.ones_like(x0)
    x, d, _ = tf.while_loop(cond, body, init)
    return x, d


def find_zero(func, x0, dx, tol, max_iter):
    def cond(x0, x1):
        f0 = func(x0)
        return tf.reduce_any(~tf.equal(x0, x1) & (tf.math.abs(f0) > tol))

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

        return x0, x1

    deriv = get_deriv_func(func)
    init = x0, x1
    return tf.while_loop(cond, body, init, maximum_iterations=max_iter)


def log_integrate(func, t0, t1, bins):
    def cond(fmax, fsum, i):
        return True

    def body(fmax, fsum, i):
        a = (2 * i + 1) / (2 * bins)
        t = (1 - a) * t0 + a * t1
        ft = func(t)
        diff = ft - fmax
        fsum = fsum + tf.math.exp(diff)
        update_fmax = diff > 0
        fsum = tf.where(update_fmax, fsum * tf.math.exp(-diff) + 1, fsum)
        fmax = tf.where(update_fmax, ft, fmax)
        return fmax, fsum, i + 1

    shape = result_shape(t0, t1)
    dtype = result_type(t0, t1)
    zero = tf.constant(0, dtype)
    bins = tf.constant(bins, dtype)
    bins = tf.where(tf.equal(t0, t1), zero, bins)
    init = tf.zeros(shape, dtype), tf.zeros(shape, dtype), zero
    fmax, fsum, _ = tf.while_loop(cond, body, init)
    h = tf.math.abs(t1 - t0) / bins
