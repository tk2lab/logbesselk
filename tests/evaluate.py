import time

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from logbesselk import log_bessel_k, log_bessel_dkdv, log_bessel_dkdx


class Timer:

    def __init__(self):
        self._start = None
        self._duration = []

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self._duration.append(time.time() - self._start)


def eval_prec(csvfile, dtype, func):
    df = pd.read_csv(csvfile)
    v = tf.constant(df['v'].astype(dtype))
    x = tf.constant(df['x'].astype(dtype))
    
    t = df['k']
    s = func(v, x).numpy()
    err_s = np.abs(s / t - 1)
    #print(err_s)
    print(err_s.mean(), err_s.std())


def eval_time(func, dtype, size, repeat):
    wrap_func = tf.function(lambda v, x: func(v, x))
    timer = Timer()
    for i in range(repeat + 1):
        v = 10 ** (3 * tf.random.uniform((size,), dtype=dtype))
        x = 10 ** (3 * tf.random.uniform((size,), dtype=dtype))
        with timer:
            wrap_func(v, x)
    print(
        timer._duration[0],
        np.mean(timer._duration[1:]),
        np.std(timer._duration[1:]),
    )


if __name__ == '__main__':

    for dtype in [np.float32, np.float64]:
        print(f'dtype: {dtype}')
        eval_prec('tests/true_log_k.csv', dtype,
            lambda v, x: tfp.math.log_bessel_kve(v, x) - x,
        )
        eval_prec('tests/true_log_k.csv', dtype, log_bessel_k)
        eval_prec('tests/true_log_dkdv.csv', dtype, log_bessel_dkdv)
        eval_prec('tests/true_log_dkdx.csv', dtype, log_bessel_dkdx)

    for dtype in [np.float32, np.float64]:
        print(f'dtype: {dtype}')
        for size in [10, 100, 1000, 10000, 100000]:
            print(f'size: {size}')
            eval_time(tfp.math.log_bessel_kve, dtype, size, 100)
            eval_time(log_bessel_k, dtype, size, 100)
            eval_time(log_bessel_dkdv, dtype, size, 100)
            eval_time(log_bessel_dkdx, dtype, size, 100)
