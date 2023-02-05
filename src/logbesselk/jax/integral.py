import jax.lax as lax
import jax.numpy as jnp
from jax import (
    grad,
)

from .math import (
    cosh,
    is_finite,
    log,
    log_cosh,
    log_sinh,
    maximum,
    square,
)
from .utils import (
    epsilon,
    extend,
    find_zero,
    log_integrate,
    result_type,
)
from .wrap import (
    wrap_bessel_ke,
    wrap_bessel_kratio,
    wrap_log_abs_deriv_bessel_k,
)

__all__ = [
    "log_bessel_k",
    "bessel_ke",
    "bessel_kratio",
    "log_abs_deriv_bessel_k",
]


def log_bessel_k(v, x):
    """
    Takashi Takekawa,
    Fast parallel calculation of modified Bessel function
    of the second kind and its derivatives,
    SoftwareX, 17, 100923 (2022).
    """
    return log_abs_deriv_bessel_k(v, x)


def bessel_kratio(v, x, d: int = 1):
    return wrap_bessel_kratio(log_bessel_k, v, x, d)


def bessel_ke(v, x):
    return wrap_bessel_ke(log_bessel_k, v, x)


@wrap_log_abs_deriv_bessel_k
def log_abs_deriv_bessel_k(v, x, m: int = 0, n: int = 0):
    def func(t):
        out = -x * cosh(t)
        if m % 2 == 0:
            out += log_cosh(v * t)
        else:
            out += log_sinh(v * t)
        if m > 0:
            out += m * log(t)
        if n > 0:
            out += n * log_cosh(t)
        return out

    def mfunc(t):
        return func(t) - th

    scale = 0.1
    tol = 1.0
    max_iter = 10
    bins = 32

    dtype = result_type(v, x)
    eps = epsilon(dtype)
    zero = dtype(0)
    scale = dtype(scale)
    bins = jnp.int32(bins)
    deriv = grad(func)

    out_is_finite = is_finite(v) & is_finite(x)
    out_is_finite &= x > 0
    if m % 2 == 1:
        out_is_finite &= v > 0

    deriv_at_zero_is_positive = out_is_finite
    if m == 0:
        deriv_at_zero_is_positive &= square(v) + m > x

    start = zero
    delta = lax.cond(deriv_at_zero_is_positive, lambda: scale, lambda: zero)
    start, delta = extend(deriv, start, delta)
    tp = find_zero(deriv, start, delta, tol, max_iter)

    th = func(tp) + log(eps) - tol
    tpl = maximum(tp - bins * eps, zero)
    tpr = maximum(tp + bins * eps, tp * (1 + bins * eps))
    mfunc_at_zero_is_negative = out_is_finite & (mfunc(zero) < 0)
    mfunc_at_tpr_is_positive = out_is_finite & (mfunc(tpr) > 0)

    start = zero
    delta = lax.cond(mfunc_at_zero_is_negative, lambda: tpl, lambda: zero)
    t0 = find_zero(mfunc, start, delta, tol, max_iter)

    start = tpr
    delta = lax.cond(mfunc_at_tpr_is_positive, lambda: scale, lambda: zero)
    start, delta = extend(mfunc, start, delta)
    t1 = find_zero(mfunc, start, delta, tol, max_iter)

    return log_integrate(func, t0, t1, bins)
