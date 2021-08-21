import tensorflow as tf

from . import math as tk
from .utils import wrap_log_k


@wrap_log_k
def log_bessel_k(v, x):
    return _log_bessel_k(v, x)[0]


def _log_bessel_k(v, x, mask=None, max_iter=100):
    """
    Digital Library of Mathematical Functions: https://dlmf.nist.gov/10.41
    """

    def local_cond(ui, next_fac, j, i, factor):
        return j >= 0

    def local_body(ui, next_fac, j, i, factor):
        k = tf.cast(i + 2 * j, x.dtype)
        ui = ui * q + factor[j]
        u0 = -factor[j] * (0.5 * k + 0.125 / (k + 1.))
        u1 =  factor[j] * (0.5 * k + 0.625 / (k + 3.))
        next_fac = tf.tensor_scatter_nd_add(next_fac, [[j], [j + 1]], [u0, u1])
        return ui, next_fac, j - 1, i, factor

    def cond(target, update, factor, i, pi):
        update = update == i
        if mask is not None:
            update &= mask
        return tf.reduce_any(update)

    def body(target, update, factor, i, pi):
        ui = tf.zeros_like(v * x)
        next_fac = tf.zeros_like(factor)
        init = ui, next_fac, i, i, factor
        ui, factor = tf.while_loop(local_cond, local_body, init)[:2]
        diff = ui / pi
        converge = tk.log(tk.abs(diff)) <= tk.log(tol) + tf.abs(const)
        target = tf.where(converge, target + diff, target)
        update = tf.where(converge, update, update + 1)
        return target, update, factor, i + 1, pi * p

    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)
    if mask is not None:
        mask = tf.convert_to_tensor(mask, tf.bool)

    tol = tk.epsilon(x.dtype)
    p = tk.sqrt(tk.square(v) + tk.square(x))
    q = tk.square(v) / (tk.square(v) + tk.square(x))

    const = 0.5 * tk.log(0.5 * tk.pi / p) + v * tk.log((v + p) / x) - p
    target = tf.zeros_like(v * x)
    update = tf.zeros_like(v * x, tf.int32)
    factor = tf.ones((max_iter,), x.dtype)
    i = tf.cast(0, tf.int32)
    pi = tf.ones_like(p)
    init = target, update, factor, i, pi
    return tf.while_loop(cond, body, init, maximum_iterations=max_iter)[:2]
