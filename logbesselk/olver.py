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
    max_iter = 100

    x = tf.convert_to_tensor(x)
    v = tf.convert_to_tensor(v, x.dtype)
    if mask is not None:
        mask = tf.convert_to_tensor(mask, tf.bool)

    p = tk.sqrt(tk.square(v) + tk.square(x))
    q = tk.square(v) / (tk.square(v) + tk.square(x))
    tol = tk.epsilon(x.dtype)
    sum_up = _olver_loop(p, q, mask, tol, max_iter)

    return (
        0.5 * tk.log(0.5 * tk.pi / p)
        + v * tk.log((v + p) / x) - p
        + tk.log(sum_up)
    )


def _olver_loop(p, q, mask, tol, max_iter):

    def get_ui_and_next_factor(factor, i):
        ui = 0.
        fac = [0.] * (i + 2)
        for j in reversed(range(i + 1)):
            ui = ui * q + factor[j]
            k = i + 2 * j
            fac[j] += factor[j] * (0.5 * k + 0.125 / (k + 1.))
            fac[j + 1] -= factor[j] * (0.5 * k + 0.625 / (k + 3.))
        return ui, fac

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
    return sum_up


def _olver_loop_(p, q, mask, tol, max_iter):

    def get_ui_and_next_factor_cond(ui, new_fac, cur_fac, i, j):
        return j >= 0

    def get_ui_and_next_factor_body(ui, new_fac, cur_fac, i, j):
        ui = ui * q + cur_fac[j]
        k = tf.cast(i + 2 * j, p.dtype)
        facj0 = cur_fac[j] * (0.5 * k + 0.125 / (k + 1.))
        facj1 = cur_fac[j] * (0.5 * k + 0.625 / (k + 3.))
        new_fac = tf.tensor_scatter_nd_add(
            new_fac, [[j], [j + 1]], [facj0, -facj1],
        )
        return ui, new_fac, cur_fac, i, j - 1

    def cond(curr, diff, pi, factor, i):
        nonzero_update = tk.abs(diff) > tol * tf.abs(curr)
        if mask is not None:
            nonzero_update &= mask
        return tf.reduce_any(nonzero_update)

    def body(curr, diff, pi, factor, i):
        ui = tf.zeros_like(p)
        ui, factor, *_ = tf.while_loop(
            get_ui_and_next_factor_cond,
            get_ui_and_next_factor_body,
            (ui, tf.zeros_like(factor), factor, i, i),
        )
        diff = ui * pi
        curr += diff
        pi /= -p
        return curr, diff, pi, factor, i + 1

    sum_up = tf.zeros_like(p)
    diff = tf.ones_like(sum_up)
    pi = tf.ones_like(p)
    factor = tf.convert_to_tensor([1.] + [0.] * max_iter, p.dtype)
    init = sum_up, diff, pi, factor, 0
    return tf.while_loop(cond, body, init, maximum_iterations=max_iter)[0]
