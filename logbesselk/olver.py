import tensorflow as tf

from . import math as tk
from .utils import wrap_log_k


@wrap_log_k
def log_bessel_k(v, x):
    return log_bessel_k(v, x)


def _log_bessel_k(v, x, mask=None):
    """
    Digital Library of Mathematical Functions: https://dlmf.nist.gov/10.41
    """

    def get_ui_and_next_factor(factor, i):
        ui = 0.
        fac = [0.] * (i + 2)
        for j in reversed(range(i + 1)):
            ui = ui * q + factor[j]
            k = i + 2 * j
            fac[j] += factor[j] * (0.5 * k + 0.125 / (k + 1.))
            fac[j + 1] -= factor[j] * (0.5 * k + 0.625 / (k + 3.))
        return ui, fac

    max_iter = 100

    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)
    if mask is not None:
        mask = tf.convert_to_tensor(mask, tf.bool)

    tol = tk.epsilon(x.dtype)
    p = tk.sqrt(tk.square(v) + tk.square(x))
    q = tk.square(v) / (tk.square(v) + tk.square(x))

    factor = [1.]
    sum_up = 0.
    for i in range(max_iter):
        ui, factor = get_ui_and_next_factor(factor, i)
        diff = ui / tk.pow(-p, i)
        sum_up += diff
        nonzero_update = tk.abs(diff) > tol * tf.abs(sum_up)
        if mask is not None:
            nonzero_update &= mask
        if tf.reduce_all(~nonzero_update):
            break

    return (
        0.5 * tk.log(0.5 * tk.pi / p)
        + v * tk.log((v + p) / x) - p
        + tk.log(sum_up)
    )
