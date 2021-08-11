import tensorflow as tf
import numpy as np

from .utils import get_deriv_func, find_zero
from . import math as tk


def log_K(v, x):
    dt0 = 1.
    n_iter = 5
    bins = 100

    dtype = (v * x).dtype
    shape = (v * x).shape
    eps = tk.epsilon(dtype)
    zero = tf.zeros(shape, dtype)
    dt0 = dt0 * tf.ones(shape, dtype)

    func = lambda t: tk.log_cosh(v * t) - x * tk.cosh(t)
    deriv1 = get_deriv_func(func)
    deriv2 = get_deriv_func(deriv1)

    t0 = tf.where(deriv2(zero) > 0., eps, zero)
    dt = tf.where(deriv1(t0) > 0., dt0, zero) 
    tp = find_zero(deriv1, t0, dt, n_iter)

    th = func(tp) + tk.log(eps)
    func_mth = lambda t: func(t) - th

    tpm = tk.maximum(tp - bins * eps, 0.)
    t0 = tf.where((func_mth(zero) < 0.) & (func_mth(tpm) > 0.),  tpm, zero)
    dt = tf.where((func_mth(zero) < 0.) & (func_mth(tpm) > 0.), -tpm, zero)
    ts = find_zero(func_mth, t0, dt, n_iter)

    t0 = tk.maximum(tp + bins * eps, tp * (1. + bins * eps))
    dt = tf.where(func_mth(t0) > 0., dt0, zero)
    te = find_zero(func_mth, t0, dt, n_iter)

    t = tf.linspace(ts, te, bins, axis=0)
    eft = tk.exp(func(t) - func(tp))
    sum_eft = tk.sum(eft, axis=0) - (eft[0] + eft[-1]) / 2
    return func(tp) + tk.log(sum_eft * (te - ts) / (bins - 1))


def log_dK_dv(v, x):
    def func(t):
        return log(t) + log_sinh(v * t) - x * cosh(t)
    t, h = _find_range(v, x, func)
    return log_sum_exp(func(t), axis=0) + log(h)


def log_minus_dK_dx(v, x):
    def func(t):
        return log_cosh(v * t) + log_cosh(t) - x * cosh(t)
    t, h = _find_range(v, x, func)
    return -log_sum_exp(func(t), axis=0) + log(h)


def dlogK_dv(v, x):
    return tf.math.exp(log_dK_dv(v, x) - log_K(v, x))


def dlogK_dx(v, x):
    return -tf.math.exp(log_minus_dK_dx(v, x) - log_K(v, x))


_tol_dt = 1e-3
_small = 1e-10
_bins = 128


def _find_range(v, x, func):
    dtype = (v * x).dtype
    tol = tf.constant(np.finfo(dtype.as_numpy_dtype).eps, dtype)
    tol_dt = tf.constant(_tol_dt, dtype)

    deriv1 = get_deriv_func(func)
    deriv2 = get_deriv_func(deriv1)

    shape = tf.shape(v * x)
    zero = tf.zeros(shape, dtype)
    one = tf.ones(shape, dtype)
    small = tf.fill(shape, tf.constant(_small, dtype))

    t0 = tf.where((deriv1(zero) == 0) & (deriv1(small) > 0), small, zero)
    dt = tf.where((deriv1(zero) > 0) | (deriv2(zero) > 0), one, zero) 
    tp = find_zero(deriv1, t0, dt, _tol_dt)

    th = func(tp) + log(tol)
    functh = lambda t: func(t) - th

    dt = tf.where(functh(zero) < 0, tp, zero)
    ts = find_zero(functh, zero, dt, _tol_dt)
    te = find_zero(functh, tp, one, _tol_dt)

    ts = tf.math.minimum(ts, tf.math.maximum(tp * (1 - _bins * tol), zero))
    te = tf.math.maximum(te, ts * (1 + 2 * _bins * tol))
    h = (te - ts) / _bins
    t = tf.linspace(ts + 0.5 * h, te - 0.5 * h, _bins)
    return t, h
