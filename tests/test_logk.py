import pytest
import tensorflow as tf
import pandas as pd
import numpy as np

from logbesselk.integral import log_bessel_k
from logbesselk.integral import bessel_ke
from logbesselk.integral import bessel_k_ratio


funcs = dict(
    logk=lambda v, x: log_bessel_k(v, x),
    log_dkdv=lambda v, x: log_bessel_k(v, x, 1, 0),
    log_dkdx=lambda v, x: log_bessel_k(v, x, 0, 1),
    ke=lambda v, x: bessel_ke(v, x),
    kratio=lambda v, x: bessel_k_ratio(v, x),
)


@pytest.mark.parametrize(
    'func, wrap, data, dtype', [
        (f, w, d, dt)
        for dt in [np.float32, np.float64]
        for d, f in funcs.items()
        for w in [False, True]
    ])
def test_logk(func, wrap, data, dtype):
    if wrap:
        func = tf.function(func)
    df = pd.read_csv(f'./data/{data}_mathematica.csv')
    v = df['v'].to_numpy().astype(dtype)
    x = df['x'].to_numpy().astype(dtype)
    val = df['true'].to_numpy().astype(dtype)
    out = func(v, x).numpy()
    assert np.allclose(out, val, rtol=5e-3, atol=0)
