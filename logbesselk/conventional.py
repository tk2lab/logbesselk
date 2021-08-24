import tensorflow as tf
from . import math as tk

from .utils import wrap_log_k
from .series import _log_bessel_ku as log_ku_small_x
from .cfraction import _log_bessel_ku as log_ku_large_x
from .utils import log_bessel_recurrence
from .asymptotic import _log_bessel_k as log_k_large_v


@wrap_log_k
def log_bessel_k(v, x):
    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)

    small_v = (tk.abs(v) < 50.) & (x > 0.)
    large_v = (tk.abs(v) >= 50.) & (x > 0.)
    small_x = small_v & (x <= 2.)
    large_x = small_v & (x > 2.)

    n = tk.round(v)
    u = v - n
    n = tf.cast(n, tf.int32)
    log_ku0_small_x, log_ku1_small_x = log_ku_small_x(u, x, small_x)
    log_ku0_large_x, log_ku1_large_x = log_ku_large_x(u, x, large_x)
    log_ku0 = tf.where(small_x, log_ku0_small_x, log_ku0_large_x)
    log_ku1 = tf.where(small_x, log_ku1_small_x, log_ku1_large_x)
    log_k = log_bessel_recurrence(log_ku0, log_ku1, u, n, x, small_v)[0]
    log_k = tf.where(large_v, log_k_large_v(v, x, large_v), log_k)
    log_k = tf.where(x < 0., tf.cast(tk.nan, x.dtype), log_k)
    log_k = tf.where(tf.equal(x, 0.), tf.cast(tk.inf, x.dtype), log_k)
    return log_k
