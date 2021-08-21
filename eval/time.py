import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm

from logbesselk.series import log_bessel_k as log_K_S
from logbesselk.cfraction import log_bessel_k as log_K_C
from logbesselk.asymptotic import _log_bessel_k as log_K_A
from logbesselk.integral import log_bessel_k as log_K_I
from logbesselk.conventional import log_bessel_k as log_K_SCA
from logbesselk.proposed import log_bessel_k as log_K_IA
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


def eval_time_log_k(dtype, n_trial):
    funcs = dict(
        S=log_K_S,
        C=log_K_C,
        I=log_K_I,
        A=log_K_A,
        SCAtfp=log_K_tfp,
        SCA=log_K_SCA,
        IA=log_K_IA,
    )

    df0 = pd.read_csv('figs/logk_prec.csv')
    #v = tf.convert_to_tensor(10 ** np.linspace(0, 2, 81) - 1, dtype)
    #x = tf.convert_to_tensor(10 ** np.linspace(-1, 2.1, 125), dtype)
    #v, x = tf.meshgrid(v, x)
    #v = tf.reshape(v, (-1,))
    #x = tf.reshape(x, (-1,))
    dfs = []
    for name, func in funcs.items():
        print(name)
        df = df0.query(f'type=="{name}"')
        v = tf.convert_to_tensor(df['v'], dtype)
        x = tf.convert_to_tensor(df['x'], dtype)
        e = tf.convert_to_tensor(df['log_err'], dtype)
        vmask = tf.boolean_mask(v, e < 4.)
        xmask = tf.boolean_mask(x, e < 4.)
        df = eval_time(func, vmask, xmask, n_trial)
        df['type'] = name
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df.to_csv('figs/logk_time.csv', index=None)


if __name__ == '__main__':
    eval_time_log_k(tf.float64, 10)
