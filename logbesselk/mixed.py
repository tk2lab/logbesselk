import tensorflow as tf
from . import math as tk

from .temme import _log_ku_temme
from .cf2 import _log_ku_cf2
from .utils import log_bessel_recurrence

from .olver import log_K as log_K_olver


def log_K(v, x):
    n = tk.round(v)
    u = v - n
    log_ku0_smallx, log_ku1_smallx = _log_ku_temme(u, x)
    log_ku0_largex, log_ku1_largex = _log_ku_cf2(u, x)
    log_ku0 = tf.where(x <= 2, log_ku0_smallx, log_ku0_largex)
    log_ku1 = tf.where(x <= 2, log_ku1_smallx, log_ku1_largex)
    log_kv_temme_cf2 = log_bessel_recurrence(log_ku0, log_ku1, u, n, x)[0]
    log_kv_olver = log_K_olver(v, x)
    return tf.where(v < 50, log_kv_temme_cf2, log_kv_olver)
