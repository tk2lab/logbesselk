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


def eval_prec(csvfile, dtype, funcs):
    funcs = {name: tf.function(func) for name, func in funcs.items()}

    df0 = pd.read_csv(csvfile)
    v = tf.convert_to_tensor(df0['v'], dtype)
    x = tf.convert_to_tensor(df0['x'], dtype)
    t = df0['true']

    dfs = []
    for name, func in funcs.items():
        k = func(v, x).numpy()
        e = (k - t) / t
        e2 = np.log10(np.abs(e) + np.finfo(e.dtype).eps)
        print(name, e2.mean(), e2.max())
        df = df0.copy()
        df['value']= k
        df['type'] = name
        df['err'] = e # / np.finfo(dtype).eps
        dfs.append(df)
    return pd.concat(dfs, axis=0)


def eval_prec_log_k(dtype):
    funcs = dict(
        tfp=log_K_tfp,
        mixed=log_K_mixed,
        #olver=log_K_olver,
        #temme=log_K_temme,
        #cf2=log_K_cf2,
        tk2=log_K_tk2,
    )
    df = eval_prec('tests/logk_grid_mathematica.csv', dtype, funcs)
    df.to_csv('tests/logk_grid_results.csv', index=None)


def eval_prec_log_k_smallv(dtype):
    funcs = dict(
        tfp=log_K_tfp,
        mixed=log_K_mixed,
        #olver=log_K_olver,
        temme=log_K_temme,
        cf2=log_K_cf2,
        tk2=log_K_tk2,
    )
    df = eval_prec('tests/logk_smallv_mathematica.csv', dtype, funcs)
    #df = eval_prec('tests/logk_mathematica.csv', dtype, funcs)
    df.to_csv('tests/logk_smallv_results.csv', index=None)


def eval_prec_dlogk_dx(dtype):
    funcs = dict(
        tfp=log_K_tfp, mixed=log_K_mixed,
        #olver=log_K_olver, temme=log_K_temme, cf2=log_K_cf2,
        tk2=log_K_tk2,
    )
    funcs = {name: dlogK_dx(func) for name, func in funcs.items()}
    eval_prec('tests/dlogk_dx_mathematica.csv', dtype, funcs)


if __name__ == '__main__':
    eval_prec_log_k(tf.float64)
    #eval_prec_log_k_smallv(np.float64)
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