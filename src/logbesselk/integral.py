import tensorflow as tf

from . import math as tk
from .utils import get_deriv_func, extend, find_zero


def log_bessel_k(v, x, n=0, m=0, name=None):

    @tf.custom_gradient
    def _custom_gradient(v, x, n, m):
        return _log_bessel_k(v, x, n, m), lambda u: _grad(n, m, u)

    def _K_grad(n, m, u):
        logkv = _custom_gradient(v, x, n, m)
        dlogkvdv = tk.exp(_custom_gradient(v, x, n + 1, m) - logkv)
        dlogkvdx = tk.exp(_custom_gradient(v, x, n, m + 1) - logkv)
        dlogkvdx = tf.where(tf.equal(m % 2, 0), -dlogkvdx, dlogkvdx)
        return u * dlogkvdv, u * dlogkvdx, None, None

    with tf.name_scope(name or 'bessel_K_tk2'):
        x = tf.convert_to_tensor(x)
        v = tf.convert_to_tensor(v, x.dtype)
        return _custom_gradient(v, x, n, m)


def _log_bessel_k(v, x, n=0, m=0,
                  mask=None, dt0=1., tol=1., max_iter=10, bins=128):

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

    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)
    if mask is None:
        mask = tf.ones_like(v * x, tf.bool)
    else:
        mask = tf.convert_to_tensor(mask, tf.bool)
    mask &= (x > 0.) & tf.where(n > 0, ~tf.equal(v, 0.), tf.cast(1, tf.bool))

    eps = tk.epsilon(x.dtype)
    zero = tf.zeros_like(v * x)
    dt0 = dt0 * tf.ones_like(v * x)
    deriv = get_deriv_func(func)

    nf = tf.cast(n, x.dtype)
    mf = tf.cast(m, x.dtype)

    out = tf.cast(tk.nan, x.dtype) # x < 0.
    out = tf.where(tf.equal(x, 0.), tf.cast(tk.inf, x.dtype), out)

    positive_peak = ~tf.equal(n, 0) | (tf.square(v) + mf > x)
    if mask is not None:
        positive_peak &= mask
    dt = tf.where(positive_peak, dt0, zero)
    ts, te = extend(deriv, zero, dt)
    tp = find_zero(deriv, te, ts, tol, max_iter)
    th = func(tp) + tk.log(eps)

    tpm = tk.maximum(tp - bins * eps, 0.)
    zero_exists = func_mth(zero) < 0.
    if mask is not None:
        zero_exists &= mask
    t0 = find_zero(func_mth, zero, tpm, tol, max_iter)

    tpp = tk.maximum(tp + bins * eps, tp * (1. + bins * eps))
    zero_exists = func_mth(tpp) > 0.
    if mask is not None:
        zero_exists &= mask
    dt = tf.where(zero_exists, dt0, zero)
    ts, te = extend(func_mth, tpp, dt)
    t1 = find_zero(func_mth, te, ts, tol, max_iter)

    t = tf.linspace(t0, t1, 2 * bins + 1, axis=0)[1:-1:2]
    h = (t1 - t0) / bins
    sum_eft = tk.sum(tk.exp(func(t) - func(tp)), axis=0)
    return tf.where(mask, func(tp) + tk.log(h * sum_eft), out)
