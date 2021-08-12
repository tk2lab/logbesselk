import tensorflow as tf
from . import math as tk

from .temme import log_K_temme, _ku_temme
from .cf2 import log_K_cf2, _ku_cf2
from .utils import bessel_recurrence

from .olver import log_K_olver

from .bessel import log_K
from .bessel import log_dK_dv
from .bessel import log_minus_dK_dx


def log_K_temme_cf2_olver(v, x):
    v = tk.abs(v)
    return tf.where(v < 50, log_K_temme_cf2(v, x), log_K_olver(v, x))


def log_K_temme_cf2(v, x):
    n = tk.round(v)
    u = v - n
    ku0_small, ku1_small = _ku_temme(u, x)
    ku0_large, ku1_large = _ku_cf2(u, x)
    ku0 = tf.where(x <= 2, ku0_small, ku0_large)
    ku1 = tf.where(x <= 2, ku1_small, ku1_large)
    return bessel_recurrence(ku0, ku1, u, x, n)[0]


def log_K_ratio_temme_cf2(v, x):
    n = tk.round(v)
    u = v - n
    ku0_small, ku1_small = _ku_temme(u, x)
    ku0_large, ku1_large = _ku_cf2(u, x)
    ku0 = tf.where(x <= 2, ku0_small, ku0_large)
    ku1 = tf.where(x <= 2, ku1_small, ku1_large)
    log_kv, log_kvp1, _ = bessel_recurrence(ku0, ku1, u, x, n)
    return log_kvp1 - _logkv
