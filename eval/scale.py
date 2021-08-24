import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm

from logbesselk.series import log_bessel_k as log_K_S
from logbesselk.cfraction import log_bessel_k as log_K_C
from logbesselk.asymptotic import _log_bessel_k as log_K_A
from logbesselk.integral import _log_bessel_k as log_K_I
from logbesselk.sca import _log_bessel_k as log_K_SCA
from .tfp import log_bessel_k as log_K_tfp

from .timer import Timer


def eval_scale(funcs, dtype, size, n_trial, suffix=''):
    funcs = {name: tf.function(func) for name, func in funcs.items()}
    for name, func in funcs.items():
        results = []
        for s in size:
            timer = Timer()
            for n in range(n_trial + 1):
                v = 10. ** (2. * tf.random.uniform((s,), dtype=dtype)) - 1.
                x = 10. ** (3. * tf.random.uniform((s,), dtype=dtype) - 1.)
                with timer:
                    func(v, x)
                if n > 0:
                    results.append([name, s, n, timer.duration[-1]])
            print(name, s, np.mean(timer.duration[1:]))
        df = pd.DataFrame(results, columns=['type', 'size', 'trial', 'time'])
        df.to_csv(f'results/logk_scale_{name}{suffix}.csv', index=None)


if __name__ == '__main__':

    funcs = dict(
        I=log_K_I,
        SCA=log_K_SCA,
        tfp=log_K_tfp,
        #I100=lambda v, x: log_K_I(v, x, max_iter=100),
    )
    size = [2 ** n for n in range(0, 18)]
    n_trial = 100

    for suffix, dtype in [('', tf.float64), ('32', tf.float32)]:
        eval_scale(funcs, dtype, size, n_trial, suffix)
