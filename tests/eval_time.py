import time

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


from logbesselk.mixed import log_bessel_k as log_K_mixed
from logbesselk.olver import log_bessel_k as log_K_olver
from logbesselk.temme import log_bessel_k as log_K_temme
from logbesselk.cf2 import log_bessel_k as log_K_cf2
from logbesselk.tk2 import log_bessel_k as log_K_tk2


def log_K_tfp(v, x):
    return tfp.math.log_bessel_kve(v, x) - x


def dlogK_dv(log_K):
    def deriv(v, x):
        with tf.GradientTape() as g:
            g.watch([v, x])
            z = log_K(v, x)
        return g.gradient(z, v)
    return deriv


def dlogK_dx(log_K):
    def deriv(v, x):
        with tf.GradientTape() as g:
            g.watch([v, x])
            z = log_K(v, x)
        return g.gradient(z, x)
    return deriv


class Timer:

    def __init__(self):
        self._start = None

    def __enter__(self):
        self._start = time.time()
        self.duration = []
        return self

    def __exit__(self, *args, **kwargs):
        self.duration.append(time.time() - self._start)


def eval_time(v, x, funcs, n_trial):
    funcs = {name: tf.function(func) for name, func in funcs.items()}
    results = []
    for name, func in funcs.items():
        func(v[0], x[0])
        sum_duration = 0.0
        for vi in v:
            for xi in x:
                timer = Timer()
                for n in range(n_trial):
                    with Timer() as timer:
                        func(vi, xi)
                results.append([name, vi.numpy(), xi.numpy(), n, np.mean(timer.duration)])
                sum_duration += np.mean(timer.duration)
        print(name, sum_duration)
    return pd.DataFrame(results, columns=['type', 'v', 'x', 'trial', 'time'])


def eval_time_log_k(dtype):
    funcs = dict(
        tk2=log_K_tk2,
        tfp=log_K_tfp,
        mixed=log_K_mixed,
        #olver=log_K_olver,
        #temme=log_K_temme,
        #cf2=log_K_cf2,
    )
    v = tf.convert_to_tensor(2. ** np.arange(-9, 10), dtype)
    x = tf.convert_to_tensor(2. ** np.arange(-9, 10), dtype)
    df = eval_time(v, x, funcs, 10)
    df.to_csv('tests/time_results.csv', index=None)


if __name__ == '__main__':
    eval_time_log_k(tf.float64)
