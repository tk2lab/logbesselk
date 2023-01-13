import pytest
import jax
import jax.numpy as jnp
import jax.lax as lax
import pandas as pd
import numpy as np

from logbesselk.jax import log_bessel_k
from logbesselk.jax import bessel_ke
from logbesselk.jax import bessel_k_ratio


funcs = dict(
    logk=log_bessel_k,
    ke=bessel_ke,
    kratio=bessel_k_ratio,
    log_dkdv=lambda v, x: log_bessel_k(v, x, 1, 0),
    log_dkdx=lambda v, x: log_bessel_k(v, x, 0, 1),
)


@pytest.mark.parametrize(
    'func, jit, data, dtype', [
        (func, jit, data, dtype)
        for dtype in [jnp.float32]
        for data, func in funcs.items()
        for jit in [False, True]
    ])
def test_logk(func, jit, data, dtype):
    df = pd.read_csv(f'./data/{data}_mathematica.csv')
    v = jnp.array(df['v'], dtype)
    x = jnp.array(df['x'], dtype)
    val = jnp.array(df['true'], dtype)
    func = jax.vmap(func)
    if jit:
        func = jax.jit(func)
    out = func(v, x)
    if jit:
        out.block_until_ready()
    assert jnp.allclose(out, val, rtol=5e-3, atol=0)
