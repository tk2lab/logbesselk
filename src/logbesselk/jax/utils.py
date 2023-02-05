import jax
import jax.lax as lax
import jax.numpy as jnp

from .math import (
    exp,
    fabs,
    log,
)

__all__ = [
    "result_type",
    "epsilon",
    "extend",
    "find_zero",
    "log_integrate",
]


def result_type(*args):
    return jnp.result_type(*args, jnp.float32).type


def epsilon(dtype):
    return jnp.finfo(dtype).eps


def extend(func, x0, dx):
    def cond(args):
        x, d, f1 = args
        return (d != 0) & (f1 > 0)

    def body(args):
        x, d, _ = args
        f1 = func(x + d)
        x, d = lax.cond(f1 > 0, lambda: (x + d, 2 * d), lambda: (x, d))
        return x, d, f1

    dummy = x0
    x, d, _ = lax.while_loop(cond, body, (x0, dx, dummy))
    return x, d


def find_zero(func, x0, dx, tol: float, max_iter: int):
    def cond(args):
        x0, x1, i = args
        f0 = func(x0)
        return (i < max_iter) & (x0 != x1) & (fabs(f0) > tol)

    def body(args):
        x0, x1, i = args
        f0 = func(x0)
        f1 = func(x1)

        x_shrink = x0 + 0.5 * (x1 - x0)
        f_shrink = func(x_shrink)
        cond = f_shrink * f0 < 0
        x1, f1 = lax.cond(cond, lambda: (x0, f0), lambda: (x1, f1))
        x0, f0 = x_shrink, f_shrink

        diff = -f0 / deriv(x0)
        ddx = diff / (x1 - x0)
        dx_in_range = (0 < ddx) & (ddx < 1)
        x_newton = lax.cond(dx_in_range, lambda: x0 + diff, lambda: x0)
        f_newton = func(x_newton)
        x1 = lax.cond(f_newton * f0 < 0, lambda: x0, lambda: x1)
        x0 = x_newton
        return x0, x1, i + 1

    deriv = jax.grad(func)
    return lax.while_loop(cond, body, (x0, x0 + dx, 0))[0]


def log_integrate(func, t0, t1, bins: int):
    def cond(args):
        fmax, fsum, i = args
        return i < bins

    def body(args):
        fmax, fsum, i = args
        a = (2 * i + 1) / (2 * bins)
        t = (1 - a) * t0 + a * t1
        ft = func(t)
        diff = ft - fmax
        fmax, fsum = lax.cond(
            diff < 0,
            lambda: (fmax, fsum + exp(diff)),
            lambda: (ft, fsum * exp(-diff) + 1),
        )
        return fmax, fsum, i + 1

    dtype = result_type(t0, t1)
    bins = lax.cond(t0 == t1, lambda: jnp.int32(0), lambda: jnp.int32(bins))
    init = dtype(0), dtype(0), jnp.int32(0)
    fmax, fsum, _ = lax.while_loop(cond, body, init)
    h = fabs(t1 - t0) / bins
    return fmax + log(fsum) + log(h)
