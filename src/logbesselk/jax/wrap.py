import functools

import jax
import jax.lax as lax
import jax.numpy as jnp

__all__ = [
    "sign_bessel_k",
    "log_bessel_recurrence",
    "wrap_simple",
    "wrap_full",
]


def sign_bessel_k(v, x, m=0, n=0):
    dtype = jnp.asarray(v * x).dtype
    plus = +jnp.ones((), dtype)
    minus = -jnp.ones((), dtype)
    if m % 2 == 0:
        if n % 2 == 0:
            return plus
        else:
            return minus
    else:
        if n % 2 == 0:
            return jnp.where(v < 0, minus, plus)
        else:
            return jnp.where(v < 0, plus, minus)


def log_bessel_recurrence(log_ku, log_kup1, u, n, x):
    def cond(args):
        ki, kj, ui, ni = args
        return ni > 0

    def body(args):
        ki, kj, ui, ni = args
        uj = ui + 1
        nj = ni - 1
        kk = jnp.logaddexp(ki, kj + jnp.log(2 * uj / x))
        k0 = jnp.where(ni > 0, kj, ki)
        k1 = jnp.where(ni > 0, kk, kj)
        return k0, k1, uj, nj

    init = log_ku, log_kup1, u, n
    return lax.while_loop(cond, body, init)[:2]


def wrap_simple(_log_bessel_k):
    @functools.wraps(_log_bessel_k)
    def log_bessel_k(v, x):
        dtype = jnp.asarray(v * x).dtype
        nan = jnp.full((), jnp.nan, dtype)
        inf = jnp.full((), jnp.inf, dtype)
        func0 = lambda: _log_bessel_k(jnp.abs(v), x)
        func1 = lambda: lax.cond(x == 0, lambda: inf, func0)
        return lax.cond(x < 0, lambda: nan, func1)

    def log_bessel_k_fwd(v, x):
        logkv = log_bessel_k(v, x)
        return logkv, (v, x, logkv)

    def log_bessel_k_bwd(res, g):
        v, x, logkv = res
        logkvm1 = log_bessel_k(v - 1, x)
        dfdx = -v / x - jnp.exp(logkvm1 - logkv)
        return None, dfdx * g

    log_bessel_k = jax.custom_vjp(log_bessel_k)
    log_bessel_k.defvjp(log_bessel_k_fwd, log_bessel_k_bwd)
    return log_bessel_k


def wrap_full(_log_bessel_k):
    @functools.wraps(_log_bessel_k)
    def log_bessel_k(v, x, m: int = 0, n: int = 0):
        if m < 0:
            raise ValueError()
        if n < 0:
            raise ValueError()
        dtype = jnp.asarray(v * x).dtype
        nan = jnp.full((), jnp.nan, dtype)
        inf = jnp.full((), jnp.inf, dtype)
        func0 = lambda: _log_bessel_k(jnp.abs(v), x, m, n)
        if m % 2 == 0:
            func1 = func0
        else:
            func1 = lambda: lax.cond(v == 0, lambda: -inf, func0)
        func2 = lambda: lax.cond(x == 0, lambda: inf, func1)
        return lax.cond(x < 0, lambda: nan, func2)

    def log_bessel_k_fwd(v, x, m, n):
        logk = log_bessel_k(v, x, m, n)
        return logk, (v, x, logk)

    def log_bessel_k_bwd(m, n, res, g):
        v, x, logk = res
        gsign = g * sign_bessel_k(v, x, m, n)
        sign_dv = sign_bessel_k(v, x, m + 1, n)
        logk_dv = log_bessel_k(v, x, m + 1, n)
        sign_dx = sign_bessel_k(v, x, m, n + 1)
        logk_dx = log_bessel_k(v, x, m, n + 1)
        log_bessel_k_dv = gsign * sign_dv * jnp.exp(logk_dv - logk)
        log_bessel_k_dx = gsign * sign_dx * jnp.exp(logk_dx - logk)
        return log_bessel_k_dv, log_bessel_k_dx

    log_bessel_k = jax.custom_vjp(log_bessel_k, nondiff_argnums=(2, 3))
    log_bessel_k.defvjp(log_bessel_k_fwd, log_bessel_k_bwd)
    return log_bessel_k
