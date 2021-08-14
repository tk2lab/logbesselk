import tensorflow as tf
from . import math as tk

from .olver import _log_K as log_K_olver
from .temme import _log_ku_temme
from .cf2 import _log_ku_cf2
from .utils import log_K_custom_gradient
from .utils import log_bessel_recurrence


def log_K(v, x, name=None):

    @tf.custom_gradient
    def custom_gradient(v, x):
        return log_K_custom_gradient(_log_K, None, None, v, x)

    with tf.name_scope(name or 'bessel_K_mix'):
        x = tf.convert_to_tensor(x)
        v = tf.convert_to_tensor(v, x.dtype)
        return custom_gradient(v, x)



def _log_K(v, x):
    mask_temme = (tk.abs(v) < 50.) & (0. < x) & (x <= 2.)
    mask_cf2 = (tk.abs(v) < 50.) & (x > 2.)
    n = tk.round(v)
    u = v - n
    n = tf.where(v < 50, n, 0.)
    log_ku0_smallx, log_ku1_smallx = _log_ku_temme(u, x, mask_temme)
    log_ku0_largex, log_ku1_largex = _log_ku_cf2(u, x, mask_cf2)
    log_ku0 = tf.where(x <= 2, log_ku0_smallx, log_ku0_largex)
    log_ku1 = tf.where(x <= 2, log_ku1_smallx, log_ku1_largex)
    log_kv_temme_cf2 = log_bessel_recurrence(log_ku0, log_ku1, u, n, x)[0]
    log_kv_olver = log_K_olver(v, x)
    return tf.where(v < 50, log_kv_temme_cf2, log_kv_olver)
