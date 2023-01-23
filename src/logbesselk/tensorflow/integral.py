import tensorflow as tf

from .math import cosh
from .math import is_finite
from .math import log
from .math import log_cosh
from .math import log_sinh
from .math import maximum
from .math import square
from .math import where
from .utils import epsilon
from .utils import extend
from .utils import find_zero
from .utils import grad
from .utils import log_integrate
from .utils import result_shape
from .utils import result_type
from .wrap import wrap_bessel_ke
from .wrap import wrap_bessel_kratio
from .wrap import wrap_log_abs_deriv_bessel_k

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
def log_abs_deriv_bessel_k(v, x, m=0, n=0):
    def func(t):
        out = -x * cosh(t)
        out += where(tf.equal(m % 2, 0), log_cosh(v * t), log_sinh(v * t))
        out = where(m > 0, out + tf.cast(m, dtype) * log(t), out)
        out = where(n > 0, out + tf.cast(n, dtype) * log_cosh(t), out)
        return out

    scale = 0.1
    tol = 1.0
    max_iter = 10
    bins = 32

    shape = result_shape(v, x)
    dtype = result_type(v, x)
    eps = epsilon(dtype)
    zero = tf.zeros(shape, dtype)
    scale = tf.constant(scale, dtype)
    deriv = grad(func)

    out_is_finite = where(
        tf.equal(m % 2, 0),
        (v >= 0) & (x > 0),
        (v > 0) & (x > 0),
    )

    deriv_at_zero_is_positive = where(
        tf.equal(m, 0),
        out_is_finite & (square(v) + tf.cast(m, dtype) > x),
        out_is_finite,
    )

    start = tf.zeros(shape, dtype)
    delta = where(deriv_at_zero_is_positive, scale, zero)
    start, delta = extend(deriv, start, delta)
    tp = find_zero(deriv, start, delta, tol, max_iter)

    th = func(tp) + log(eps) - tol
    mfunc = lambda t: func(t) - th
    tpl = maximum(tp - bins * eps, zero)
    tpr = maximum(tp + bins * eps, tp * (1 + bins * eps))
    mfunc_at_zero_is_negative = out_is_finite & (mfunc(zero) < 0)
    mfunc_at_tpr_is_positive = out_is_finite & (mfunc(tpr) > 0)

    start = zero
    delta = where(mfunc_at_zero_is_negative, tpl, zero)
    t0 = find_zero(mfunc, start, delta, tol, max_iter)

    start = tpr
    delta = where(mfunc_at_tpr_is_positive, scale, zero)
    start, delta = extend(mfunc, start, delta)
    t1 = find_zero(mfunc, start, delta, tol, max_iter)

    return log_integrate(func, t0, t1, bins)
