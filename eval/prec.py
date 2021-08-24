import tensorflow as tf
import pandas as pd
import numpy as np

from logbesselk.series import log_bessel_k as log_K_S
from logbesselk.cfraction import log_bessel_k as log_K_C
from logbesselk.asymptotic import _log_bessel_k as log_K_A
from logbesselk.integral import _log_bessel_k as log_K_I
from logbesselk.sca import log_bessel_k as log_K_SCA
from logbesselk.utils import get_deriv_func
from .tfp import log_bessel_k as log_K_tfp


def eval_prec_local(func, v, x, t):
    k = func(v, x)
    if k is not None:
        k = k.numpy()
        v = v.numpy()
        x = x.numpy()
        e = (k - t) / t
        e2 = np.log10(np.abs(e) / np.finfo(e.dtype).eps + 1)
        return pd.DataFrame(dict(v=v, x=x, true=t, value=k, err=e, log_err=e2))


def eval_prec(kind, funcs, dtype):
    df0 = pd.read_csv(f'data/{kind}_mathematica.csv')
    v = tf.convert_to_tensor(df0['v'], dtype)
    x = tf.convert_to_tensor(df0['x'], dtype)
    t = df0['true']

    for name, func in funcs.items():
        print(kind, name)
        df = eval_prec_local(func, v, x, t)
        if df is not None:
            df.to_csv(f'results/{kind}_prec_{name}.csv', index=None)
            print(df['log_err'].min(), df['log_err'].mean(), df['log_err'].max())


if __name__ == '__main__':
    dtype = tf.float64
    funcs = dict(
        I10=lambda v, x: log_K_I(v, x, max_iter=10),
        A30=lambda v, x: log_K_A(v, x, max_iter=30),
        #S=log_K_S,
        #C=log_K_C,
        SCA=log_K_SCA,
        #tfp=log_K_tfp,
    )
    eval_prec('logk', funcs, dtype)

    dv_funcs = {name: get_deriv_func(func, 0) for name, func in funcs.items()}
    eval_prec('dlogk_dv', dv_funcs, dtype)

    dx_funcs = {name: get_deriv_func(func, 1) for name, func in funcs.items()}
    eval_prec('dlogk_dx', dx_funcs, dtype)
