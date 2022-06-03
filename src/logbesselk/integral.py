import tensorflow as tf

from . import math as tk
from .utils import find_peak
from .utils import find_zero
from .utils import find_zero_with_extend

__all__ = [
    "sign_bessel_k",
    "log_bessel_k",
    "slog_bessel_k",
    "bessel_ke",
    "bessel_k_ratio",
]


def sign_bessel_k(v, x, m=0, n=0, name=None):
    with tf.name_scope(name or "sign_bessel_k_tk2"):
        dtype = tk.common_dtype([v, x], tf.float32)
        x = tf.convert_to_tensor(x, dtype)
        v = tf.convert_to_tensor(v, dtype)
        return _sign_bessel_k_naive(v, x, m, n)


def log_bessel_k(v, x, m=0, n=0, name=None):
    with tf.name_scope(name or "log_bessel_k_tk2"):
        dtype = tk.common_dtype([v, x], tf.float32)
        x = tf.convert_to_tensor(x, dtype)
        v = tf.convert_to_tensor(v, dtype)
        return _log_bessel_k_custom_gradient(v, x, m, n)


def slog_bessel_k(v, x, m=0, n=0, name=None):
    with tf.name_scope(name or "slog_bessel_k_tk2"):
        dtype = tk.common_dtype([v, x], tf.float32)
        x = tf.convert_to_tensor(x, dtype)
        v = tf.convert_to_tensor(v, dtype)
        sign = _sign_bessel_k_naive(v, x, m, n)
        logk = _log_bessel_k_custom_gradient(v, x, m, n)
        return sign, logk


def bessel_ke(v, x, m=0, n=0, name=None):
    with tf.name_scope(name or "bessel_ke_tk2"):
        dtype = tk.common_dtype([v, x], tf.float32)
        x = tf.convert_to_tensor(x, dtype)
        v = tf.convert_to_tensor(v, dtype)
        sign = _sign_bessel_k_naive(v, x, m, n)
        logk = _log_bessel_k_custom_gradient(v, x, m, n)
        return sign * tk.exp(logk + x)


def bessel_k_ratio(v, x, d=1, m=0, n=0, name=None):
    with tf.name_scope(name or "bessel_k_ratio_tk2"):
        dtype = tk.common_dtype([v, x], tf.float32)
        x = tf.convert_to_tensor(x, dtype)
        v = tf.convert_to_tensor(v, dtype)
        d = tf.convert_to_tensor(d, dtype)
        signd = _sign_bessel_k_naive(v + d, x, m, n)
        logkd = _log_bessel_k_custom_gradient(v + d, x, m, n)
        sign = _sign_bessel_k_naive(v, x, m, n)
        logk = _log_bessel_k_custom_gradient(v, x, m, n)
        return signd * sign * tf.exp(logkd - logk)


def _log_bessel_k_custom_gradient(v, x, m, n):
    @tf.custom_gradient
    def logk_vx(v, x):
        def _grad(u):
            sign = _sign_bessel_k_naive(v, x, m, n)
            sign_dv = _sign_bessel_k_naive(v, x, m + 1, n)
            logk_dv = _log_bessel_k_custom_gradient(v, x, m + 1, n)
            sign_dx = _sign_bessel_k_naive(v, x, m, n + 1)
            logk_dx = _log_bessel_k_custom_gradient(v, x, m, n + 1)
            return (
                u * sign_dv * sign * tk.exp(logk_dv - logk),
                u * sign_dx * sign * tk.exp(logk_dx - logk),
            )

        logk = _log_bessel_k_naive(v, x, m, n)
        return logk, _grad

    return logk_vx(v, x)


def _sign_bessel_k_naive(v, x, m, n):
    dtype = tk.common_dtype([v, x])
    if m % 2 == 0:
        return tf.cast(1 if n % 2 == 0 else -1, dtype)
    elif n % 2 == 0:
        return tf.where(v < 0, tf.cast(-1, dtype), tf.cast(1, dtype))
    else:
        return tf.where(v < 0, tf.cast(1, dtype), tf.cast(-1, dtype))


def _log_bessel_k_func(v, x, t, m, n):
    out = tk.log_cosh(v * t) if m % 2 == 0 else tk.log_sinh(v * t)
    out -= x * tk.cosh(t)
    if m > 0:
        out += m * tk.log(t)
    if n > 0:
        out += n * tk.log_cosh(t)
    return out


def _log_bessel_k_naive(
    v, x, m=0, n=0, mask=None, dt0=0.1, tol=1.0, max_iter=10, bins=128
):
    def func(t):
        return _log_bessel_k_func(v, x, t, m, n)

    def func_mth(t):
        return func(t) - th

    v = tk.abs(v)

    dtype = tk.common_dtype([v, x])
    shape = tf.shape(v * x)

    condzero = tf.zeros(shape, tf.bool) if m % 2 == 0 else tf.equal(v, 0)
    condinf = tf.equal(x, 0)
    condnan = x < 0

    if mask is None:
        mask = tf.ones(shape, tf.bool)
    mask &= ~condzero & ~condinf & ~condnan

    eps = tk.epsilon(dtype)
    zero = tf.zeros(shape, dtype)
    scale = tf.cast(dt0, dtype)

    zero_peak = tf.zeros(shape, tf.bool) if m > 0 else tf.square(v) + m < x
    dt = tf.where(~zero_peak & mask, scale, zero)
    tp = find_peak(func, zero, dt, tol, max_iter)
    th = func(tp) + tk.log(eps) - tol

    tpm = tp - bins * eps
    tpm_positive = tpm > 0
    tpm = tf.where(tpm_positive & mask, tpm, zero)
    t0 = find_zero(func_mth, zero, tpm, tol, max_iter)

    tpp = tk.maximum(tp + bins * eps, tp * (1 + bins * eps))
    zero_exists = func_mth(tpp) > 0
    dt = tf.where(zero_exists & mask, scale, zero)
    t1 = find_zero_with_extend(func_mth, tpp, dt, tol, max_iter)

    t = tf.linspace(t0, t1, 2 * bins + 1, axis=0)[1:-1:2]
    h = (t1 - t0) / bins
    out = tk.log(h) + tk.log_sum_exp(func(t), axis=0)

    out = tf.where(condzero, tf.cast(-tk.inf, dtype), out)
    out = tf.where(condinf, tf.cast(tk.inf, dtype), out)
    out = tf.where(condnan, tf.cast(tk.nan, dtype), out)
    return out
