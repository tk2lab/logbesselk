import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm

from logbesselk.series import log_bessel_k as log_K_S
from logbesselk.cfraction import log_bessel_k as log_K_C
from logbesselk.asymptotic import _log_bessel_k as log_K_A
from logbesselk.integral import _log_bessel_k as log_K_I
from logbesselk.sca import log_bessel_k as log_K_SCA
from .tfp import log_bessel_k as log_K_tfp

from .timer import Timer


def eval_time(func, v, x, n_trial):
    func = tf.function(func)
    func(v[0], x[0])
    results = []
    timer = Timer()
    with tqdm(total=v.numpy().size * n_trial) as pbar:
        for vi, xi in zip(v, x):
            for n in range(n_trial):
                with Timer() as timer:
                    func(vi, xi)
                pbar.update()
            results.append([
                vi.numpy(), xi.numpy(), n,
                np.mean(timer.duration[-n_trial:]),
            ])
    return pd.DataFrame(results, columns=['v', 'x', 'trial', 'time'])


def eval_time_log_k(name, func, dtype, n_trial, suffix):
    df = pd.read_csv(f'results/logk_prec_{name}{suffix}.csv')
    v = tf.convert_to_tensor(df['v'], dtype)
    x = tf.convert_to_tensor(df['x'], dtype)
    e = tf.convert_to_tensor(df['log_err'], dtype)
    vmask = tf.boolean_mask(v, e < 4.)
    xmask = tf.boolean_mask(x, e < 4.)
    df = eval_time(func, vmask, xmask, n_trial)
    df.to_csv(f'results/logk_time_{name}{suffix}.csv', index=None)


if __name__ == '__main__':

    n_trial = 10

    funcs = dict(
        I=log_K_I,
        A=log_K_A,
        S=log_K_S,
        C=log_K_C,
        SCA=log_K_SCA,
        tfp=log_K_tfp,
        #A10=lambda v, x: log_K_A(v, x, max_iter=10),
        #I100=lambda v, x: log_K_I(v, x, max_iter=100),
    )

    for suffix, dtype in [('', tf.float64), ('32', tf.float32)]:
        for name, func in funcs.items():
            print('log_k', name)
            eval_time_log_k(name, func, dtype, n_trial, suffix)
