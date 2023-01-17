import jax
import jax.lax as lax
import jax.numpy as jnp

from .math import log_cosh
from .math import log_sinh
from .misc import wrap_bessel_k_ratio
from .misc import wrap_bessel_ke
from .misc import wrap_slog_bessel_k
from .utils import extend
from .utils import find_zero
from .utils import log_integrate
from .wrap import wrap_full

__all__ = [
    "log_bessel_k",
    "bessel_k_ratio",
    "bessel_ke",
    "slog_bessel_k",
]


@wrap_full
def log_bessel_k(v, x, m: int = 0, n: int = 0):
    """
    Takashi Takekawa,
    Fast parallel calculation of modified Bessel function
    of the second kind and its derivatives,
    SoftwareX, 17, 100923 (2022).
    """

    def func(t):
        if m % 2 == 0:
            out = log_cosh(v * t) - x * jnp.cosh(t)
        else:
            out = log_sinh(v * t) - x * jnp.cosh(t)
        if m > 0:
            out += m * jnp.log(t)
        if n > 0:
            out += n * log_cosh(t)
        return out

    tol = 1.0
    bins = 32
    scale = 0.1
    max_iter = 10

    dtype = jnp.result_type(v, x)
    eps = jnp.finfo(dtype).eps

    deriv = jax.grad(func)
    tol = jnp.asarray(tol, dtype)
    bins = jnp.asarray(bins, dtype)

    zero = jnp.asarray(0, dtype)
    scale = jnp.asarray(scale, dtype)

    v = jnp.asarray(v, dtype)
    x = jnp.asarray(x, dtype)

    if m % 2 == 1:
        out_is_finite = (v > 0) & (x > 0)
    else:
        out_is_finite = x > 0

    if m == 0:
        deriv_at_zero_is_positive = out_is_finite & (jnp.square(v) + m >= x)
    else:
        deriv_at_zero_is_positive = out_is_finite

    start = zero
    delta = lax.cond(deriv_at_zero_is_positive, lambda: scale, lambda: zero)
    start, dt = extend(deriv, start, delta)
    tp = find_zero(deriv, start, delta, tol, max_iter)

    th = func(tp) + jnp.log(eps) - tol
    mfunc = lambda t: func(t) - th
    tpl = jnp.maximum(tp - bins * eps, zero)
    tpr = jnp.maximum(tp + bins * eps, tp * (1 + bins * eps))
    mfunc_at_zero_is_negative = out_is_finite & (mfunc(zero) <= 0)
    mfunc_at_tpr_is_positive = out_is_finite & (mfunc(tpr) <= 0)

    start = zero
    delta = lax.cond(mfunc_at_zero_is_negative, lambda: tpl, lambda: zero)
    t0 = find_zero(mfunc, start, delta, tol, max_iter)

    start = tpr
    delta = lax.cond(mfunc_at_tpr_is_positive, lambda: scale, lambda: zero)
    start, delta = extend(mfunc, start, delta)
    t1 = find_zero(mfunc, start, delta, tol, max_iter)

    return log_integrate(func, t0, t1, bins)


def slog_bessel_k(v, x, m=0, n=0):
    return wrap_slog_bessel_k(log_bessel_k, v, x, m, n)


def bessel_k_ratio(v, x, d=1):
    return wrap_bessel_k_ratio(log_bessel_k, v, x, d)


def bessel_ke(v, x):
    return wrap_bessel_ke(log_bessel_k, v, x)
