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
        self._duration = []

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args, **kwargs):
        self._duration.append(time.time() - self._start)


def eval_prec(csvfile, dtype, funcs):
    #pd.options.display.float_format = '{:.6f}'.format
    #funcs = {name: tf.function(func) for name, func in funcs.items()}

    df0 = pd.read_csv(csvfile)
    v = tf.constant(df0['v'].astype(dtype))
    x = tf.constant(df0['x'].astype(dtype))
    t = df0['true']

    df = df0.copy()
    for name, func in funcs.items():
        k = func(v, x).numpy()
        e = (k - t) / t
        print(name, np.abs(e).mean())
        df[name] = k
        df[f'err_{name}'] = e # / np.finfo(dtype).eps
    return df
    #print(df)
    #print(np.log10(np.abs(df.iloc[:, 3:]).mean()))
    #print(np.log10(np.abs(df.iloc[:, 3:]).max()))

    '''
    df = df0.copy()
    for name, func in funcs.items():
        with Timer() as timer:
            for i in range(10):
                func(v, x)
        df[f'{name}'] = np.mean(timer._duration)
    print(np.abs(df.iloc[:, 3:]).mean())
    '''


def eval_prec_log_k_smallv(dtype):
    funcs = dict(
        tfp=log_K_tfp,
        mixed=log_K_mixed,
        #olver=log_K_olver,
        #temme=log_K_temme,
        #cf2=log_K_cf2,
        tk2=log_K_tk2,
    )
    df = eval_prec('tests/logk_smallv_mathematica.csv', dtype, funcs)
    #df = eval_prec('tests/logk_mathematica.csv', dtype, funcs)
    df.to_csv('tests/logk_smallv_results.csv')


def eval_prec_log_k(dtype):
    funcs = dict(
        tfp=log_K_tfp,
        mixed=log_K_mixed,
        olver=log_K_olver,
        temme=log_K_temme,
        cf2=log_K_cf2,
        tk2=log_K_tk2,
    )
    eval_prec('tests/logk_mathematica.csv', dtype, funcs)


def eval_prec_dlogk_dx(dtype):
    funcs = dict(
        tfp=log_K_tfp, mixed=log_K_mixed,
        #olver=log_K_olver, temme=log_K_temme, cf2=log_K_cf2,
        tk2=log_K_tk2,
    )
    funcs = {name: dlogK_dx(func) for name, func in funcs.items()}
    eval_prec('tests/dlogk_dx_mathematica.csv', dtype, funcs)


if __name__ == '__main__':
    eval_prec_log_k_smallv(np.float64)
    #eval_prec_dlogk_dx(np.float64)

    '''
    for dtype in [np.float32, np.float64]:
        print(f'dtype: {dtype}')

        if True:
            eval_prec('tests/true_log_k.csv', dtype,
                lambda v, x: tfp.math.log_bessel_kve(v, x) - x,
            )
            eval_prec('tests/true_log_k.csv', dtype, log_bessel_k)
            eval_prec('tests/true_log_dkdv.csv', dtype, log_bessel_dkdv)
            eval_prec('tests/true_log_dkdx.csv', dtype, log_bessel_dkdx)

        if True:
            print(f'dtype: {dtype}')
            for size in [10, 100, 1000, 10000, 100000]:
                print(f'size: {size}')
                eval_time(tfp.math.log_bessel_kve, dtype, size, 100)
                eval_time(log_bessel_k, dtype, size, 100)
                eval_time(log_bessel_dkdv, dtype, size, 100)
                eval_time(log_bessel_dkdx, dtype, size, 100)
    '''
