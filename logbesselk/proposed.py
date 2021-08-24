import tensorflow as tf
from . import math as tk

from .utils import wrap_log_k
from .integral import _log_bessel_k as log_k_small_v
from .asymptotic import _log_bessel_k as log_k_large_v


@wrap_log_k
def log_bessel_k(v, x):
    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)

    small_v = (tk.abs(v) < 50.) & (x > 0.)
    large_v = (tk.abs(v) >= 50.) & (x > 0.)

    log_k = tf.cast(tk.nan, x.dtype) # x < 0.
    log_k = tf.where(tf.equal(x, 0.), tf.cast(tk.inf, x.dtype), log_k)
    log_k = tf.where(small_v, log_k_small_v(v, x, mask=small_v), log_k)
    log_k = tf.where(large_v, log_k_large_v(v, x, mask=large_v), log_k)
    return log_k
