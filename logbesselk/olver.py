import tensorflow as tf
import numpy as np

from . import math as tk


def log_K(v, x, name=None):
    """
    Digital Library of Mathematical Functions: https://dlmf.nist.gov/10.41
    """

    def _next_factor(prev, k):
        fac = [0.] * (3 * k + 4)
        for i in range(k, 3 * k + 1, 2):
            fac[i + 1] += prev[i] * (0.5 * i + 0.125 / (i + 1.))
            fac[i + 3] -= prev[i] * (0.5 * i + 0.625 / (i + 3.))
        return fac

    def _u(factor, k, q):
        uk = 0.
        for fac in reversed(factor):
            uk = uk / q + fac
        return uk

    n_factors = 10

    with tf.name_scope(name or 'olver_log_K'):
        v_abs = tk.abs(v)
        q = tk.sqrt(1. + tk.square(x / v_abs))

        sum_uv = 0.
        factor = [1.]
        for k in range(n_factors):
            sum_uv += _u(factor, k, q) / tk.pow(-v_abs, k)
            factor = _next_factor(factor, k)

        return (
            0.5 * tk.log(0.5 * np.pi / (v_abs * q))
            + v_abs * tk.log((v_abs + v_abs * q) / x)
            - v_abs * q
            + tf.math.log(sum_uv)
        )
