import tensorflow as tf

from . import math as tk
from .asymptotic import _log_bessel_k as log_k_large_v
from .cfraction import _log_bessel_ku as log_ku_large_x
from .series import _log_bessel_ku as log_ku_small_x
from .utils import log_bessel_recurrence
from .utils import wrap_log_k

__all__ = [
    "log_bessel_k",
]


@wrap_log_k
def log_bessel_k(v, x):
    return _log_bessel_k(v, x)


def _log_bessel_k(v, x, return_counter=False):
    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)

    v = tk.abs(v)
    n = tk.round(v)
    u = v - n

    small_v_ = v < 25
    small_x_ = x < 1.6 + (1 / 2) * tk.log(v + 1)

    small_x = small_x_ & small_v_ & (x > 0)
    large_x = ~small_x_ & small_v_
    small_v = small_v_ & (x > 0)
    large_v = ~small_v_ & (x > 0)

    lk0s, lk1s, cs = log_ku_small_x(u, x, small_x, return_counter=True)
    lk0l, lk1l, cl = log_ku_large_x(u, x, large_x, return_counter=True)
    lk0 = tf.where(small_x, lk0s, lk0l)
    lk1 = tf.where(small_x, lk1s, lk1l)
    out_small_v = log_bessel_recurrence(lk0, lk1, u, n, x, small_v)[0]
    out_large_v, cv = log_k_large_v(v, x, large_v, return_counter=True)

    out = tf.cast(tk.nan, x.dtype)  # x < 0.
    out = tf.where(tf.equal(x, 0), tf.cast(tk.inf, x.dtype), out)
    out = tf.where(small_v, out_small_v, out)
    out = tf.where(large_v, out_large_v, out)

    if return_counter:
        return out, (cs, cl, cv)
    return out
