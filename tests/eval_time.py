import tensorflow as tf
import pandas as pd
import numpy as np

from logbesselk.series import log_bessel_k as log_K_S
from logbesselk.cfraction import log_bessel_k as log_K_C
from logbesselk.asymptotic import log_bessel_k as log_K_A
from logbesselk.integral import log_bessel_k as log_K_I
from logbesselk.conventional import log_bessel_k as log_K_SCA
from logbesselk.proposed import log_bessel_k as log_K_IA
from .tfp import log_bessel_k as log_K_tfp

from .timer import Timer


def eval_time(v, x, funcs, n_trial):
    funcs = {name: tf.function(func) for name, func in funcs.items()}
    results = []
    for name, func in funcs.items():
        print(name)
        func(v[0], x[0])
        sum_duration = 0.0
        for vi in v:
            for xi in x:
                timer = Timer()
                for n in range(n_trial):
                    with Timer() as timer:
                        func(vi, xi)
                results.append([
                    name, vi.numpy(), xi.numpy(), n, np.mean(timer.duration),
                ])
                sum_duration += np.mean(timer.duration)
            print('*', end='', flush=True)
        print()
        print(sum_duration)
    return pd.DataFrame(results, columns=['type', 'v', 'x', 'trial', 'time'])


def eval_time_log_k(dtype):
    funcs = dict(
        A=log_K_A,
        C=log_K_C,
        S=log_K_S,
        SCA=log_K_SCA,
        SCAtfp=log_K_tfp,
        I=log_K_I,
        IA=log_K_IA,
    )
    v = tf.convert_to_tensor(10 ** np.linspace(0, 2, 81, dtype) - 1)
    x = tf.convert_to_tensor(10 ** np.linspace(-1, 2.1, 125), dtype)
    #v = tf.convert_to_tensor(10 ** np.linspace(0, 2, 11, dtype) - 1)
    #x = tf.convert_to_tensor(10 ** np.linspace(-1, 2.1, 11), dtype)
    df = eval_time(v, x, funcs, 10)
    df.to_csv('tests/time_results.csv', index=None)


if __name__ == '__main__':
    eval_time_log_k(tf.float64)
