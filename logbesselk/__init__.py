import tensorflow as tf
from . import math as tk

from .bessel import log_K
from .bessel import log_dK_dv
from .bessel import log_minus_dK_dx
from .temme import log_K_temme, log_K_temme_original, log_K_temme_cambell
from .olver import log_K_olver


def log_K_temme_olver(v, x):
    v = tk.abs(v)
    return tf.where(v < 50, log_K_temme(v, x), log_K_olver(v, x))
