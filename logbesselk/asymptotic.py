import tensorflow as tf
import numpy as np

from . import math as tk
from .utils import wrap_log_k


@wrap_log_k
def log_bessel_k(v, x):
    return _log_bessel_k(v, x)


def _log_bessel_k(v, x, mask=None, max_iter=30, return_counter=False):
    """
    Digital Library of Mathematical Functions: https://dlmf.nist.gov/10.41
    """

    def cond(target, i, pi, update):
        if mask is not None:
            update &= mask
        return tf.reduce_any(update)

    def body(target, i, pi, update):

        def local_cond(ui, j):
            return j >= 0

        def local_body(ui, j):
            ui = ui * q + factor[i, j]
            return ui, j - 1

        ui = tf.zeros_like(v * x)
        ui, _ = tf.while_loop(local_cond, local_body, (ui, i))
        target = tf.where(update, target + ui / pi, target)
        update &= tk.abs(ui / pi) > tol * tk.abs(target)
        return target, i + 1, pi * p, update

    factor = np.zeros((max_iter, max_iter))
    factor[0, 0] = 1.
    for i in range(max_iter - 1):
        for j in range(i + 1):
            k = i + 2. * j
            factor[i + 1, j + 0] -= factor[i, j] * (0.5 * k + 0.125 / (k + 1.))
            factor[i + 1, j + 1] += factor[i, j] * (0.5 * k + 0.625 / (k + 3.))

    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)
    if mask is not None:
        mask = tf.convert_to_tensor(mask, tf.bool)

    tol = tk.epsilon(x.dtype)
    factor = tf.convert_to_tensor(factor, x.dtype)
    p = tk.sqrt(tk.square(v) + tk.square(x))
    q = tk.square(v) / (tk.square(v) + tk.square(x))

    log_const = 0.5 * tk.log(0.5 * tk.pi / p) + v * tk.log((v + p) / x) - p
    target = tf.zeros_like(v * x)
    i = tf.cast(0, tf.int32)
    pi = tf.ones_like(p)
    update = tf.ones_like(v * x, tf.bool)
    init = target, i, pi, update
    target, counter, *_ = tf.while_loop(
        cond, body, init, maximum_iterations=max_iter,
    )
    results = log_const + tk.log(target)

    if return_counter:
        return results, i
    return results
