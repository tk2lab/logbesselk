import tensorflow as tf
import pandas as pd
import numpy as np

from logbesselk.series import log_bessel_k as log_K_S
from logbesselk.cfraction import log_bessel_k as log_K_C
from logbesselk.asymptotic import log_bessel_k as log_K_A
from logbesselk.integral import _log_bessel_k as log_K_I
from logbesselk.conventional import log_bessel_k as log_K_SCA
from logbesselk.proposed import log_bessel_k as log_K_IA
from .tfp import log_bessel_k as log_K_tfp


def eval_prec(func, v, x, t):
    k = func(v, x).numpy()
    v = v.numpy()
    x = x.numpy()
    e = (k - t) / t
    e2 = np.log10(np.abs(e) / np.finfo(e.dtype).eps + 1)
    return pd.DataFrame(dict(v=v, x=x, true=t, value=k, err=e, log_err=e2))


def eval_prec_log_k(funcs, dtype):
    df0 = pd.read_csv('data/logk_mathematica.csv')
    v = tf.convert_to_tensor(df0['v'], dtype)
    x = tf.convert_to_tensor(df0['x'], dtype)
    t = df0['true']

    for name, func in funcs.items():
        print(name)
        df = eval_prec(func, v, x, t)
        df.to_csv(f'data/logk_prec_{name}.csv', index=None)
        print(df['log_err'].mean(), df['log_err'].max())


if __name__ == '__main__':
    dtype = tf.float64
    funcs = dict(
        #A=log_K_A,
        #S=log_K_S,
        #C=log_K_C,
        #I=log_K_I,
        #tfp=log_K_tfp,
        #I10=lambda v, x: log_K_I(v, x, max_iter=10),
        I20=lambda v, x: log_K_I(v, x, max_iter=20),
    )
    eval_prec_log_k(funcs, dtype)
