import jax
import jax.lax as lax
import jax.numpy as jnp

__all__ = [
    "extend",
    "find_zero",
    "log_integrate",
]


def extend(func, x0, dx):
    def cond(args):
        x, d, f1 = args
        return jnp.any((d != 0) & (f1 > 0))

    def body(args):
        x, d, _ = args
        f1 = func(x + d)
        x = jnp.where(f1 > 0, x + d, x)
        d = jnp.where(f1 > 0, 2 * d, d)
        return x, d, f1

    dummy = jnp.ones_like(x0)
    x, d, _ = lax.while_loop(cond, body, (x0, dx, dummy))
    return x, d


def find_zero(func, x0, dx, tol, max_iter):
    def cond(args):
        x0, x1, i = args
        f0 = func(x0)
        return (i < max_iter) & jnp.any((x0 != x1) & (jnp.abs(f0) > tol))

    def body(args):
        x0, x1, i = args
        f0 = func(x0)
        f1 = func(x1)

        x_shrink = x0 + 0.5 * (x1 - x0)
        f_shrink = func(x_shrink)
        x1, f1 = lax.cond(
            f_shrink * f0 < 0,
            lambda: (x0, f0),
            lambda: (x1, f1),
        )
        x0, f0 = x_shrink, f_shrink

        dx = x1 - x0
        diff = -f0 / deriv(x0)
        ddx = diff / dx
        cond = (0 < ddx) & (ddx < 1)
        x_newton = lax.cond(cond, lambda: x0 + diff, lambda: x0)
        f_newton = func(x_newton)
        x1 = lax.cond(f_newton * f0 < 0, lambda: x0, lambda: x1)
        x0 = x_newton
        return x0, x1, i + 1

    deriv = jax.grad(func)
    return lax.while_loop(cond, body, (x0, x0 + dx, 0))[0]


def log_integrate(func, t0, t1, bins):
    def funcb(b):
        a = (2 * b + 1) / (2 * bins)
        t = (1 - a) * t0 + a * t1
        return func(t)

    def cond(args):
        fmax, fsum, b = args
        return b <= bins

    def body(args):
        fmax, fsum, b = args
        ft = funcb(b)
        diff = ft - fmax
        cond = diff < 0
        fmax, fsum = lax.cond(
            diff < 0,
            lambda: (fmax, fsum + jnp.exp(diff)),
            lambda: (ft, fsum * jnp.exp(-diff) + 1),
        )
        return fmax, fsum, b + 1

    fmax = funcb(0)
    fsum = jnp.ones((), fmax.dtype)
    fmax, fsum, _ = lax.while_loop(cond, body, (fmax, fsum, 1))
    h = jnp.abs(t1 - t0) / bins
    return fmax + jnp.log(fsum) + jnp.log(h)
