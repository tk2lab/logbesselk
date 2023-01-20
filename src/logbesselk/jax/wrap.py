import functools

import jax
import jax.lax as lax
import jax.numpy as jnp

__all__ = [
    "wrap_log_bessel_k",
    "wrap_log_abs_deriv_bessel_k",
    "wrap_bessel_ke",
    "wrap_bessel_kratio",
]


def wrap_log_bessel_k(func):
    @functools.wraps(func)
    def wrapped_func(v, x):
        dtype = jnp.result_type(v, x, jnp.float32).type
        v = jnp.abs(v)
        return lax.cond(
            x > 0,
            lambda: func(v, x),
            lambda: lax.cond(
                x < 0,
                lambda: dtype(jnp.nan),
                lambda: dtype(jnp.inf),
            ),
        )

    def fwd(v, x):
        out = func(v, x)
        return out, (v, x, out)

    def bwd(res, g):
        v, x, out = res
        outm1 = custom_func(v - 1, x)
        dx = -v / x - jnp.exp(outm1 - out)
        return None, dx * g

    custom_func = jax.custom_vjp(wrapped_func)
    custom_func.defvjp(fwd, bwd)
    return custom_func


def wrap_log_abs_deriv_bessel_k(func):
    @functools.wraps(func)
    def wrapped_func(v, x, m: int = 0, n: int = 0):
        if m < 0:
            raise ValueError()
        if n < 0:
            raise ValueError()
        dtype = jnp.result_type(v, x, jnp.float32).type
        v = jnp.abs(v)
        return lax.cond(
            (x > 0) if m % 2 == 0 else (x > 0) & (v > 0),
            lambda: func(v, x, m, n),
            lambda: lax.cond(
                x < 0,
                lambda: dtype(jnp.nan),
                lambda: lax.cond(
                    x == 0,
                    lambda: dtype(jnp.inf),
                    lambda: dtype(-jnp.inf),
                ),
            ),
        )

    def fwd(v, x, m, n):
        out = custom_func(v, x, m, n)
        return out, (v, x, out)

    def bwd(m, n, res, g):
        v, x, out = res
        dv = jnp.sign(v) * jnp.exp(custom_func(v, x, m + 1, n) - out)
        dx = -jnp.exp(custom_func(v, x, m, n + 1) - out)
        return dv * g, dx * g

    custom_func = jax.custom_vjp(wrapped_func, nondiff_argnums=(2, 3))
    custom_func.defvjp(fwd, bwd)
    return custom_func


def wrap_bessel_ke(log_bessel_k, v, x):
    logk = log_bessel_k(v, x)
    return jnp.exp(logk + x)


def wrap_bessel_kratio(log_bessel_k, v, x, d=1):
    logk = log_bessel_k(v, x)
    logkd = log_bessel_k(v + d, x)
    return jnp.exp(logkd - logk)
