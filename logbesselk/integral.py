import tensorflow as tf

from . import math as tk
from .utils import get_deriv_func, extend, find_zero


def log_bessel_k(v, x, name=None):

    @tf.custom_gradient
    def _log_K_custom_gradient(v, x, n, m):
        return _log_bessel_k(v, x, n, m), lambda u: _log_K_grad(n, m, u)

    def _log_K_grad(n, m, u):
        logkv = _log_K_custom_gradient(v, x, n, m)
        dlogkvdv = tk.exp(_log_K_custom_gradient(v, x, n + 1, m) - logkv)
        dlogkvdx = tk.exp(_log_K_custom_gradient(v, x, n, m + 1) - logkv)
        dlogkvdx = tf.where(tf.equal(m % 2, 0), -dlogkvdx, dlogkvdx)
        return u * dlogkvdv, u * dlogkvdx

    with tf.name_scope(name or 'bessel_K_tk2'):
        x = tf.convert_to_tensor(x)
        v = tf.convert_to_tensor(v, x.dtype)
        return _log_K_custom_gradient(v, x, 0, 0)


def _log_bessel_k(
        v, x, n=0, m=0, mask=None, dt0=1., tol=1., n_iter=100, bins=128):

    def func(t):
        out = tf.where(
            tf.equal(n % 2, 0), tk.log_cosh(v * t), tk.log_sinh(v * t),
        )
        out -= x * tk.cosh(t)
        out += tf.where(n > 0, nf * tk.log(t), tf.cast(0., x.dtype))
        out += tf.where(m > 0, mf * tk.log_cosh(t), tf.cast(0., x.dtype))
        return out

    def func_mth(t):
        return func(t) - th

    dtype = (v * x).dtype
    shape = (v * x).shape

    eps = tk.epsilon(dtype)
    zero = tf.zeros(shape, dtype)
    dt0 = dt0 * tf.ones(shape, dtype)
    deriv = get_deriv_func(func)

    nf = tf.cast(n, x.dtype)
    mf = tf.cast(m, x.dtype)

    positive_peak = ~tf.equal(n, 0) | (tf.square(v) + mf > x)
    if mask is not None:
        positive_peak &= mask
    dt = tf.where(positive_peak, dt0, zero)
    ts, te = extend(deriv, zero, dt)
    tp = find_zero(deriv, ts, te, tol, n_iter)
    th = func(tp) + tk.log(eps)

    tpm = tk.maximum(tp - bins * eps, 0.)
    zero_exists = func_mth(zero) < 0.
    if mask is not None:
        zero_exists &= mask
    ts = tf.where(zero_exists,  tpm, zero)
    ts, te = extend(func_mth, ts, -ts)
    t0 = find_zero(func_mth, ts, te, tol, n_iter)

    tpp = tk.maximum(tp + bins * eps, tp * (1. + bins * eps))
    zero_exists = func_mth(tpp) > 0.
    if mask is not None:
        zero_exists &= mask
    dt = tf.where(zero_exists, dt0, zero)
    ts, te = extend(func_mth, tpp, dt)
    t1 = find_zero(func_mth, ts, te, tol, n_iter)

    t = tf.linspace(t0, t1, bins + 1, axis=0)
    eft = tk.exp(func(t) - func(tp))
    sum_eft = tk.sum(eft, axis=0) - 0.5 * (eft[0] + eft[-1])
    return func(tp) + tk.log(sum_eft * (t1 - t0) / bins)
