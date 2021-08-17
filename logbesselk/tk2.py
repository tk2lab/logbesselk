import tensorflow as tf

from . import math as tk
from .utils import get_deriv_func, extend, find_zero


def log_bessel_k(v, x, name=None):

    @tf.custom_gradient
    def _log_K_custom_gradient(v, x, n, m):
        return _log_K(v, x, n, m), lambda u: _log_K_grad(n, m, u)

    def _log_K_grad(n, m, u):
        logkv = _log_K_custom_gradient(v, x, n, m)
        dlogkvdv = tk.exp(_log_K_custom_gradient(v, x, n + 1, m) - logkv)
        dlogkvdx = tk.exp(_log_K_custom_gradient(v, x, n, m + 1) - logkv)
        if m % 2 == 0:
            dlogkvdx *= -1
        return u * dlogkvdv, u * dlogkvdx

    with tf.name_scope(name or 'bessel_K_tk2'):
        x = tf.convert_to_tensor(x)
        v = tf.convert_to_tensor(v, x.dtype)
        return _log_K_custom_gradient(v, x, n=0, m=0)


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
    deriv = get_deriv_func(func)

    dt = dt0
    if n == 0:
        dt = tf.where(tf.square(v) + m <= x, zero, dt)
    t0, t1 = extend(deriv, zero, dt)
    tp = find_zero(deriv, t0, t1, n_iter)
    th = func(tp) + tk.log(eps)

    tpm = tk.maximum(tp - bins * eps, 0.)
    no_zero = func_mth(zero) >= 0.
    t0 = tf.where(no_zero, zero,  tpm)
    dt = tf.where(no_zero, zero, -tpm)
    t0, t1 = extend(func_mth, t0, dt)
    ts = find_zero(func_mth, t0, t1, n_iter)

    t0 = tk.maximum(tp + bins * eps, tp * (1. + bins * eps))
    dt = tf.where(func_mth(t0) > 0., dt0, zero)
    t0, t1 = extend(func_mth, t0, dt)
    te = find_zero(func_mth, t0, t1, n_iter)

    t = tf.linspace(ts, te, bins + 1, axis=0)
    eft = tk.exp(func(t) - func(tp))
    sum_eft = tk.sum(eft, axis=0) - 0.5 * (eft[0] + eft[-1])
    return func(tp) + tk.log(sum_eft * (te - ts) / bins)
