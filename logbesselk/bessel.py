import tensorflow as tf
import numpy as np

from .utils import get_deriv_func, find_zero
from .math import abs, sign, log, cosh, log_sinh, log_cosh, log_sum_exp


def log_bessel_k(v, x):
    def deriv0(t):
        return log_cosh(v * t) - x * cosh(t)
    return _log_bessel_common(v, x, deriv0)


def log_bessel_dkdv(v, x):
    def deriv0(t):
        return log(t) + log_sinh(v * t) - x * cosh(t)
    return sign(v) * _log_bessel_common(abs(v), x, deriv0)


def log_bessel_dkdx(v, x):
    def deriv0(t):
        return log_cosh(v * t) + log_cosh(t) - x * cosh(t)
    return _log_bessel_common(v, x, deriv0)


def _log_bessel_common(v, x, deriv0):

    def derivth(t):
        return deriv0(t) - th

    shape = tf.shape(v * x)
    dtype = (v * x).dtype

    tol = tf.constant(np.finfo(dtype.as_numpy_dtype).eps, dtype)
    tol_dt = tf.constant(1e-5, dtype)
    max_iter = 100
    bins = 256

    zero = tf.zeros(shape, dtype)
    one = tf.ones(shape, dtype)

    deriv1 = get_deriv_func(deriv0)
    deriv2 = get_deriv_func(deriv1)

    dt = tf.where((deriv1(zero) > 0) | (deriv2(zero) > 0), one, zero) 
    tp = find_zero(deriv1, zero, dt, tol_dt, max_iter)
    th = deriv0(tp) + log(tol)
    te = find_zero(derivth, tp, one, tol_dt, max_iter)

    h = te / bins
    t = tf.linspace(0.5 * h, te - 0.5 * h, bins)
    return log_sum_exp(deriv0(t), axis=0) + log(h)
