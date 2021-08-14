import tensorflow as tf
import numpy as np

from . import math as tk
from .utils import get_deriv_func, find_zero


def log_K(v, x, name=None):
    with tf.name_scope(name or 'bessel_K_tk2'):
        x = tf.convert_to_tensor(x)
        v = tf.convert_to_tensor(v, x.dtype)
        return _log_K_custom_gradient(v, x, n=0, m=0)


@tf.custom_gradient
def _log_K_custom_gradient(v, x, n, m):
    return _log_K(v, x, n, m), lambda u: _log_K_grad(v, x, n, m, u)


def _log_K_grad(v, x, n, m, u):
    logkv = _log_K_custom_gradient(v, x, n, m)
    dlogkvdv = tk.exp(_log_K_custom_gradient(v, x, n + 1, m) - logkv)
    dlogkvdx = tk.exp(_log_K_custom_gradient(v, x, n, m + 1) - logkv)
    if m % 2 == 0:
        dlogkvdx *= -1
    return u * dlogkvdv, u * dlogkvdx


def _log_K(v, x, n, m, dt0=1., n_iter=5, bins=128):

    def func(t):
        out = - x * tk.cosh(t)
        if n > 0:
            out += n * tk.log(t)
        if m > 0:
            out += m * tk.log_cosh(t)
        if n % 2 == 0:
            out += tk.log_cosh(v * t)
        else:
            out += tk.log_sinh(v * t)
        return out

    def func_mth(t):
        return func(t) - th

    dtype = (v * x).dtype
    shape = (v * x).shape

    eps = tk.epsilon(dtype)
    zero = tf.zeros(shape, dtype)
    dt0 = dt0 * tf.ones(shape, dtype)
    deriv1 = get_deriv_func(func)

    if n > 0:
        t0 = zero
        dt = dt0
    else:
        deriv2 = get_deriv_func(deriv1)
        t0 = tf.where(deriv2(zero) > 0., eps, zero)
        dt = tf.where(deriv1(t0) > 0., dt0, zero) 
    tp = find_zero(deriv1, t0, dt, n_iter)
    th = func(tp) + tk.log(eps)

    tpm = tk.maximum(tp - bins * eps, 0.)
    if n > 0:
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