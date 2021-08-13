import tensorflow as tf
import numpy as np

from .utils import get_deriv_func, find_zero
from . import math as tk


def log_K(v, x):
    func = lambda t: tk.log_cosh(v * t) - x * tk.cosh(t)
    return _common_integral(func, v, x, kind=1, dt0=1., n_iter=5, bins=128)


def log_dK_dv(v, x):
    func = lambda t: log(t) + log_sinh(v * t) - x * cosh(t)
    return _common_integral(func, v, x, kind=0, dt0=1., n_iter=5, bins=128)


def log_minus_dK_dx(v, x):
    func = lambda t: log_cosh(v * t) + log_cosh(t) - x * cosh(t)
    return -_common_integral(func, v, x, kind=1, dt0=1., n_iter=5, bins=128)


def dlogK_dv(v, x):
    return tf.math.exp(log_dK_dv_integral(v, x) - log_K(v, x))


def dlogK_dx(v, x):
    return -tf.math.exp(log_minus_dK_dx(v, x) - log_K(v, x))


def _common_integral(func, v, x, kind, dt0=1., n_iter=5, bins=128):
    dtype = (v * x).dtype
    shape = (v * x).shape
    eps = tk.epsilon(dtype)
    zero = tf.zeros(shape, dtype)
    dt0 = dt0 * tf.ones(shape, dtype)
    deriv1 = get_deriv_func(func)

    if kind == 1:
        deriv2 = get_deriv_func(deriv1)
        t0 = tf.where(deriv2(zero) > 0., eps, zero)
        dt = tf.where(deriv1(t0) > 0., dt0, zero) 
        tp = find_zero(deriv1, t0, dt, n_iter)
        th = func(tp) + tk.log(eps)
        func_mth = lambda t: func(t) - th
    else:
        t0 = zero
        dt = dt0
    tp = find_zero(deriv1, zero, dt, n_iter)
    th = func(tp) + tk.log(eps)
    func_mth = lambda t: func(t) - th

    tpm = tk.maximum(tp - bins * eps, 0.)
    if kind == 1:
        t0 = tf.where((func_mth(zero) < 0.) & (func_mth(tpm) > 0.),  tpm, zero)
        dt = tf.where((func_mth(zero) < 0.) & (func_mth(tpm) > 0.), -tpm, zero)
    else:
        t0 = tpm
        dt = tf.where(func_mth(tpm) > 0., -tpm, zero)
    ts = find_zero(func_mth, t0, dt, n_iter)

    t0 = tk.maximum(tp + bins * eps, tp * (1. + bins * eps))
    dt = tf.where(func_mth(t0) > 0., dt0, zero)
    te = find_zero(func_mth, t0, dt, n_iter)

    t = tf.linspace(ts, te, bins + 1, axis=0)
    eft = tk.exp(func(t) - func(tp))
    sum_eft = tk.sum(eft, axis=0) - 0.5 * (eft[0] + eft[-1])
    return func(tp) + tk.log(sum_eft * (te - ts) / bins)
