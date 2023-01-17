import os
os.environ["JAX_ENABLE_X64"] = "True"

import pytest
import jax
import jax.numpy as jnp
import jax.lax as lax
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.special import kve as scipy_kve

from logbesselk.jax.integral import bessel_ke as int_kve
from logbesselk.jax.sca import bessel_ke as sca_kve


int_kve_jit = lambda v, x: jax.jit(int_kve)(v, x).block_until_ready()
int_kve_vmap = jax.vmap(int_kve)
int_kve_vmap_jit = lambda v, x: jax.jit(int_kve_vmap)(v, x).block_until_ready()

sca_kve_jit = lambda v, x: jax.jit(sca_kve)(v, x).block_until_ready()
sca_kve_vmap = jax.vmap(sca_kve)
sca_kve_vmap_jit = lambda v, x: jax.jit(sca_kve_vmap)(v, x).block_until_ready()

vi = np.array([-10, -2, -1, 0, 1, 2, 10])
vf = st.uniform(-10, 20).rvs(2)
vlist = np.concatenate([vi, vf], 0)
print(vlist)

xf = st.uniform(0, 50).rvs(2) - 50
xlist = np.concatenate([[-1, 0], xf], 0)
print(xlist)


@pytest.mark.parametrize(
    'func, dtype, tol, v, x', [
        (func, dtype, tol, v, x)
        for func in [sca_kve_jit, int_kve_jit]
        for dtype, tol in [(np.float32, 1e-6), (np.float64, 1e-10)]
        for v in vlist
        for x in xlist
    ])
def test_logk(func, dtype, tol, v, x):
    v = v.astype(dtype)
    x = x.astype(dtype)
    out = func(v, x)
    scipy_out = scipy_kve(v, x)
    np.testing.assert_allclose(out, scipy_out, rtol=tol)


@pytest.mark.parametrize(
    'func, dtype, tol', [
        (func, dtype, tol)
        for func in [sca_kve_vmap_jit, int_kve_vmap_jit]
        for dtype, tol in [(np.float32, 1e-6), (np.float64, 1e-10)]
    ])
def test_logk_vmap(func, dtype, tol):
    v = vlist.astype(dtype)
    x = xlist.astype(dtype)
    v, x = np.meshgrid(v, x)
    v, x = v.ravel(), x.ravel()
    out = func(v, x)
    scipy_out = scipy_kve(v, x)
    np.testing.assert_allclose(out, scipy_out, rtol=tol)
