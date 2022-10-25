import pytest
import tensorflow as tf
import pandas as pd
import numpy as np

from logbesselk.integral import log_bessel_k
from logbesselk.integral import bessel_ke
from logbesselk.integral import bessel_k_ratio


funcs = dict(
    logk=lambda v, x, bins: log_bessel_k(v, x, bins=bins),
    log_dkdv=lambda v, x, bins: log_bessel_k(v, x, 1, 0, bins=bins),
    log_dkdx=lambda v, x, bins: log_bessel_k(v, x, 0, 1, bins=bins),
    ke=lambda v, x, bins: bessel_ke(v, x, bins=bins),
    kratio=lambda v, x, bins: bessel_k_ratio(v, x, bins=bins),
)


@pytest.mark.parametrize(
    'func, wrap, data, dtype, bins', [
        (f, w, d, dt, bins)
        for bins in [32]
        for dt in [np.float32, np.float64]
        for d, f in funcs.items()
        for w in [False, True]
    ])
def test_logk(func, wrap, data, dtype, bins):
    func_bins = lambda v, x: func(v, x, bins)
    if wrap:
        wrap_func = tf.function(func_bins)
    else:
        wrap_func = func_bins
    df = pd.read_csv(f'./data/{data}_mathematica.csv')
    v = df['v'].to_numpy().astype(dtype)
    x = df['x'].to_numpy().astype(dtype)
    val = df['true'].to_numpy().astype(dtype)
    out = wrap_func(v, x).numpy()
    assert np.allclose(out, val, rtol=5e-3, atol=0)
