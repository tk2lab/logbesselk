import tensorflow as tf
import numpy as np

from .utils import get_deriv_func, find_zero
from .math import abs, sign, log, cosh, log_sinh, log_cosh, log_sum_exp


def log_bessel_k(v, x, m=100, tol_dt=1e-5, tol=None, max_iter=10):
    def deriv0(t):
        return log_cosh(v * t) - x * cosh(t)
    return _log_bessel_common(v, x, deriv0, m, tol_dt, tol, max_iter)


def log_bessel_dkdv(v, x, m=100, tol_dt=1e-5, tol=None, max_iter=10):
    def deriv0(t):
        return log(t) + log_sinh(v * t) - x * cosh(t)
    return sign(v) * _log_bessel_common(
        abs(v), x, deriv0, m, tol_dt, tol, max_iter,
    )


def log_bessel_dkdx(v, x, m=100, tol_dt=1e-5, tol=None, max_iter=10):
    def deriv0(t):
        return log_cosh(v * t) + log_cosh(t) - x * cosh(t)
    return _log_bessel_common(v, x, deriv0, m, tol_dt, tol, max_iter)


def _log_bessel_common(v, x, deriv0, m, tol_dt, tol, max_iter):

    def derivth(t):
        return deriv0(t) - th

    shape = tf.shape(v * x)
    dtype = (v * x).dtype

    if tol is None:
        tol = np.finfo(dtype.as_numpy_dtype).eps

    tol_dt = tf.constant(tol_dt, dtype)
    tol = tf.constant(tol, dtype)
    m = tf.constant(m)

    zero = tf.zeros(shape, dtype)
    one = tf.ones(shape, dtype)

    deriv1 = get_deriv_func(deriv0)
    deriv2 = get_deriv_func(deriv1)

    t0 = tf.fill(shape, tol_dt)
    dt = tf.where(deriv1(t0) > 0, one, zero) 
    tp = find_zero(deriv1, t0, dt, tol_dt, max_iter)
    th = deriv0(tp) + log(tol)

    dt = tf.where(derivth(zero) < 0, tp, zero)
    ts = find_zero(derivth, dt, -dt, tol_dt, max_iter)
    te = find_zero(derivth, tp, one, tol_dt, max_iter)

    h = (te - ts) / tf.cast(m, dtype)
    t = tf.linspace(ts + 0.5 * h, te - 0.5 * h, m)
    return log_sum_exp(deriv0(t), axis=0) + log(h)
