import tensorflow as tf
import numpy as np

from . import math as tk
from .utils import get_deriv_func, find_zero
from .utils import wrap_K, wrap_log_K


def K(v, x, name=None):
    return wrap_K(
        _log_K, _log_dKdv, _log_minus_dKdx, v, x, name or 'bessel_K_int',
    )


def log_K(v, x, name=None):
    return wrap_log_K(
        _log_K, _log_dKdv, _log_minus_dKdx, v, x, name or 'log_K_int',
    )



def _log_K(v, x):
    def _log_f(t):
        return tk.log_cosh(v * t) - x * tk.cosh(t)
    return _common_integral(_log_f, v, x, kind=1)


def _log_dKdv(v, x):
    def _log_dfdv(t):
        return tk.log(t) + tk.log_cosh(v * t) - x * tk.cosh(t)
    return _common_integral(_log_dfdv, v, x, kind=0)


def _log_minus_dKdx(v, x):
    def _log_minus_dfdx(t):
        return tk.log_cosh(t) + tk.log_cosh(v * t) - x * tk.cosh(t)
    return _common_integral(_log_minus_dfdx, v, x, kind=1)


def _common_integral(func, v, x, kind, dt0=1., n_iter=5, bins=128):
    dtype = (v * x).dtype
    shape = (v * x).shape

    eps = tk.epsilon(dtype)
    zero = tf.zeros(shape, dtype)
    dt0 = dt0 * tf.ones(shape, dtype)
    deriv1 = get_deriv_func(func)

    if kind == 0:
        t0 = zero
        dt = dt0
    else:
        deriv2 = get_deriv_func(deriv1)
        t0 = tf.where(deriv2(zero) > 0., eps, zero)
        dt = tf.where(deriv1(t0) > 0., dt0, zero) 
    tp = find_zero(deriv1, t0, dt, n_iter)
    th = func(tp) + tk.log(eps)
    func_mth = lambda t: func(t) - th

    tpm = tk.maximum(tp - bins * eps, 0.)
    if kind == 0:
        t0 = tpm
        dt = tf.where(func_mth(tpm) > 0., -tpm, zero)
    else:
        t0 = tf.where((func_mth(zero) < 0.) & (func_mth(tpm) > 0.),  tpm, zero)
        dt = tf.where((func_mth(zero) < 0.) & (func_mth(tpm) > 0.), -tpm, zero)
    ts = find_zero(func_mth, t0, dt, n_iter)

    t0 = tk.maximum(tp + bins * eps, tp * (1. + bins * eps))
    dt = tf.where(func_mth(t0) > 0., dt0, zero)
    te = find_zero(func_mth, t0, dt, n_iter)

    t = tf.linspace(ts, te, bins + 1, axis=0)
    eft = tk.exp(func(t) - func(tp))
    sum_eft = tk.sum(eft, axis=0) - 0.5 * (eft[0] + eft[-1])
    return func(tp) + tk.log(sum_eft * (te - ts) / bins)
