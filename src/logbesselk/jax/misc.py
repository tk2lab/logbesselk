import jax.lax as lax
import jax.numpy as jnp

__all__ = [
    "log_bessel_recurrence",
    "sign_bessel_k_deriv",
]


def sign_bessel_k_deriv(v, x, m=0, n=0):
    dtype = jnp.result_type(v, x)
    sign = jnp.asarray(1, dtype)
    if n % 2 == 1:
        sign * -1
    if m % 2 == 0:
        return sign
    else:
        return jnp.sign(v) * sign


def log_bessel_recurrence(log_ku, log_kup1, u, n, x):
    def cond(args):
        ki, kj, ui, ni = args
        return ni > 0

    def body(args):
        ki, kj, ui, ni = args
        uj = ui + 1
        nj = ni - 1
        kk = jnp.logaddexp(ki, kj + jnp.log(2 * uj / x))
        return kj, kk, uj, nj

    init = log_ku, log_kup1, u, n
    return lax.while_loop(cond, body, init)[:2]
