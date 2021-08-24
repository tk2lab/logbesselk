import tensorflow as tf
import pandas as pd
import numpy as np

import logbesselk.math as tk
from logbesselk.series import log_bessel_k as log_K_S
from logbesselk.cfraction import log_bessel_k as log_K_C
from logbesselk.asymptotic import _log_bessel_k as log_K_A
from logbesselk.integral import _log_bessel_k as _log_K_I
from logbesselk.integral import log_bessel_k as log_K_I
from logbesselk.sca import log_bessel_k as log_K_SCA
from logbesselk.utils import get_deriv_func
from .tfp import log_bessel_k as log_K_tfp


def eval_prec_local(func, v, x, t):
    k = func(v, x)
    eps = np.finfo(x.dtype.as_numpy_dtype()).eps
    if k is not None:
        k = k.numpy()
        v = v.numpy()
        x = x.numpy()
        e = (k - t) / t
        e2 = np.log10(np.abs(e) / eps + 1)
        return pd.DataFrame(dict(v=v, x=x, true=t, value=k, err=e, log_err=e2))


def eval_prec(kind, funcs, dtype, suffix=''):
    df0 = pd.read_csv(f'data/{kind}_mathematica.csv')
    v = tf.convert_to_tensor(df0['v'], dtype)
    x = tf.convert_to_tensor(df0['x'], dtype)
    t = df0['true']

    for name, func in funcs.items():
        print(kind, name)
        df = eval_prec_local(func, v, x, t)
        if df is not None:
            df.to_csv(f'results/{kind}_prec_{name}{suffix}.csv', index=None)
            print(df['log_err'].min(), df['log_err'].mean(), df['log_err'].max())


if __name__ == '__main__':

    funcs = dict(
        I=log_K_I,
        A=log_K_A
        S=log_K_S,
        C=log_K_C,
        SCA=log_K_SCA,
        tfp=log_K_tfp,
        #I100=lambda v, x: _log_K_I(v, x, max_iter=100),
        #A10=lambda v, x: log_K_A(v, x, max_iter=10),
    )
    dv_funcs = dict(
        I=lambda v, x: _log_K_I(v, x, n=1),
        #I100=lambda v, x: _log_K_I(v, x, n=1, max_iter=100),
    )
    dx_funcs = dict(
        I=lambda v, x: _log_K_I(v, x, m=1),
        SCA=lambda v, x: tk.log_add_exp(log_K_SCA(v - 1, x), tk.log(v / x) + log_K_SCA(v, x)),
        tfp=lambda v, x: tk.log_add_exp(log_K_tfp(v - 1, x), tk.log(v / x) + log_K_tfp(v, x)),
        #I100=lambda v, x: _log_K_I(v, x, m=1, max_iter=100),
    )

    for suffix, dtype in [('', tf.float64), ('32', tf.float32)]:
        eval_prec('logk', funcs, dtype, suffix)
        eval_prec('log_dkdv', dv_funcs, dtype, suffix)
        eval_prec('log_dkdx', dx_funcs, dtype, suffix)
