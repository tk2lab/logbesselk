import jax
import jax.lax as lax
import jax.numpy as jnp

from .math import log_cosh
from .math import log_sinh
from .utils import extend
from .utils import find_zero
from .utils import log_integrate
from .wrap import wrap_full

__all__ = [
    "log_bessel_k",
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

    dt0 = 0.1
    tol = 1.0
    bins = 32
    max_iter = 10

    dtype = jnp.asarray(v * x).dtype
    eps = jnp.finfo(dtype).eps
    zero = jnp.zeros((), dtype)
    scale = jnp.full((), dt0, dtype)
    deriv = jax.grad(func)

    start = zero
    if m == 0:
        dt = scale
    else:
        dt = lax.cond(jnp.square(v) + m >= x, lambda: scale, lambda: zero)
    start, dt = extend(deriv, start, dt)
    tp = find_zero(deriv, start, dt, tol, max_iter)

    th = func(tp) + jnp.log(eps) - tol
    func_mth = lambda t: func(t) - th

    start = zero
    dt = jnp.maximum(tp - bins * eps, 0)
    t0 = find_zero(func_mth, start, dt, tol, max_iter)

    start = jnp.maximum(tp + bins * eps, tp * (1 + bins * eps))
    dt = lax.cond(func_mth(start) > 0, lambda: scale, lambda: zero)
    start, dt = extend(func_mth, start, dt)
    t1 = find_zero(func_mth, start, dt, tol, max_iter)

    return log_integrate(func, t0, t1, bins)
