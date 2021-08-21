import tensorflow as tf
from . import math as tk

from .utils import wrap_log_k
from .olver import _log_bessel_k as log_k_olver
from .temme import _log_bessel_ku as log_ku_temme
from .cf2 import _log_bessel_ku as log_ku___cf2
from .utils import log_bessel_recurrence


@wrap_log_k
def log_bessel_k(v, x):
    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)

    small_v = (tk.abs(v) < 50.) & (x > 0.)
    large_v = (tk.abs(v) >= 50.) & (x > 0.)
    small_x = small_v & (x <= 2.)
    large_x = small_v & (x > 2.)

    log_k = tf.cast(tk.nan, x.dtype) # x < 0.
    log_k = tf.where(tf.equal(x, 0.), tf.cast(tk.inf, x.dtype), log_k)
    log_k = tf.where(large_v, log_k_olver(v, x, large_v), log_k)

    n = tk.round(v)
    u = v - n
    n = tf.cast(n, tf.int32)
    log_ku0_temme, log_ku1_temme = log_ku_temme(u, x, small_x)
    log_ku0___cf2, log_ku1___cf2 = log_ku___cf2(u, x, large_x)
    log_ku0 = tf.where(small_x, log_ku0_temme, log_ku0___cf2)
    log_ku1 = tf.where(small_x, log_ku1_temme, log_ku1___cf2)
    log_k_smallv = log_bessel_recurrence(log_ku0, log_ku1, u, n, x, small_v)[0]
    return tf.where(small_v, log_k_smallv, log_k)
