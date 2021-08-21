import tensorflow as tf
import pandas as pd
import numpy as np


from logbesselk.mixed import log_bessel_k as log_K_mixed
from logbesselk.olver import log_bessel_k as log_K_olver
from logbesselk.temme import log_bessel_k as log_K_temme
from logbesselk.cf2 import log_bessel_k as log_K_cf2
from logbesselk.tk2 import log_bessel_k as log_K_tk2
from .tfp import log_bessel_k as log_K_tfp
from .timer import Timer


def eval_scale(size, funcs, dtype, n_trial):
    funcs = {name: tf.function(func) for name, func in funcs.items()}
    results = []
    for name, func in funcs.items():
        for s in size:
            timer = Timer()
            for n in range(n_trial + 1):
                v = 100. * tf.random.uniform((s,), dtype=dtype)
                x = 100. * tf.random.uniform((s,), dtype=dtype)
                with timer:
                    func(v, x)
                if n > 0:
                    results.append([name, s, timer.duration[-1]])
            print(name, s, np.mean(timer.duration[1:]))
    return pd.DataFrame(results, columns=['type', 'size', 'time'])


def eval_scale_log_k(dtype):
    funcs = dict(
        tk2=log_K_tk2,
        mixed=log_K_mixed,
        tfp=log_K_tfp,
        #olver=log_K_olver,
        #temme=log_K_temme,
        #cf2=log_K_cf2,
    )
    df = eval_scale([2 ** n for n in range(16)], funcs, dtype, 10)
    df.to_csv('tests/scale_results.csv', index=None)


if __name__ == '__main__':
    eval_scale_log_k(tf.float64)
