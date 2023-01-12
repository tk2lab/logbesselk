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
        x0, x1 = x_shrink, jnp.where(f_shrink * f0 < 0, x0, x1)
        f0, f1 = f_shrink, jnp.where(f_shrink * f0 < 0, f0, f1)

        dx = x1 - x0
        diff = -f0 / deriv(x0)
        ddx = diff / dx
        x_newton = jnp.where((0 < ddx) & (ddx < 1), x0 + diff, x0)
        f_newton = func(x_newton)
        x0, x1 = x_newton, jnp.where(f_newton * f0 < 0, x0, x1)
        return x0, x1, i + 1

    deriv = jax.grad(func)
    return lax.while_loop(cond, body, (x0, x0 + dx, 0))[0]


def log_integrate(func, t0, t1, bins):
    def funcb(b):
        a = (2 * b + 1) / (2 * bins)
        t = (1 - a) * t0 + a * t1
        return func(t)

    def cond(args):
        fsum, fmax, b = args
        return b <= bins

    def body(args):
        fsum, fmax, b = args
        ft = funcb(b)
        diff = ft - fmax
        cond = diff < 0
        fsum = jnp.where(cond, fsum + jnp.exp(diff), fsum * jnp.exp(-diff) + 1)
        fmax = jnp.where(cond, fmax, ft)
        return fsum, fmax, b + 1

    init = jnp.ones_like(t0), funcb(0), 1
    fsum, fmax, _ = lax.while_loop(cond, body, init)
    h = (t1 - t0) / bins
    return fmax + jnp.log(fsum) + jnp.log(h)
