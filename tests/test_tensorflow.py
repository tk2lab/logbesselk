import functools
import os

import pytest
import tensorflow as tf
import numpy as np

from logbesselk.tensorflow.utils import grad

from logbesselk.tensorflow.sca import log_bessel_k as logk_sca
from logbesselk.tensorflow.integral import log_bessel_k as logk_int

dlogkdv_int = grad(logk_int, 0)
dlogkdx_sca = grad(logk_sca, 1)
dlogkdx_int = grad(logk_int, 1)

from logbesselk.tensorflow.sca import bessel_ke as ke_sca
from logbesselk.tensorflow.integral import bessel_ke as ke_int

from logbesselk.tensorflow.integral import log_abs_deriv_bessel_k as logdk_int

logdkdv_int = lambda v, x: logdk_int(v, x, 1, 0)
logdkdx_int = lambda v, x: logdk_int(v, x, 0, 1)

tol = 2e-5


def gen_func_fixture(func, kind_list, jit_list):
    def gen_func(request):
        kind, jit = request.param
        base = globals()[f"{func}_{kind}"]
        if jit == "jit":
            base = tf.function(base)
        @functools.wraps(base)
        def out_func(*args, **kwargs):
            return base(*args, **kwargs).numpy()
        return out_func
    return pytest.fixture(
        scope="session",
        params=[
            (kind, jit)
            for kind in kind_list
            for jit in jit_list
        ], ids=lambda p: f"{p[0]}-{p[1]}"
    )(gen_func)


logk_func = gen_func_fixture("logk", ["sca", "int"], ["jit"])


def test_logk(logk_func, logk_data):
    v, x, ans = logk_data
    out = logk_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


@pytest.mark.vec()
def test_logk_vec(logk_func, logk_vec_data):
    v, x, ans = logk_vec_data
    out = logk_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


ke_func = gen_func_fixture("ke", ["sca", "int"], ["jit"])


def test_ke(ke_func, ke_data):
    v, x, ans = ke_data
    out = ke_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


@pytest.mark.vec()
def test_ke_vec(ke_func, ke_vec_data):
    v, x, ans = ke_vec_data
    out = ke_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


dlogkdx_func = gen_func_fixture("dlogkdx", ["sca", "int"], ["jit"])


def test_dlogkdx(dlogkdx_func, dlogkdx_data):
    v, x, ans = dlogkdx_data
    out = dlogkdx_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


@pytest.mark.vec()
def test_dlogkdx_vec(dlogkdx_func, dlogkdx_vec_data, dtype):
    v, x, ans = dlogkdx_vec_data
    out = dlogkdx_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


dlogkdx_func = gen_func_fixture("dlogkdx", ["sca", "int"], ["jit"])


def test_dlogkdx(dlogkdx_func, dlogkdx_data):
    v, x, ans = dlogkdx_data
    out = dlogkdx_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


@pytest.mark.vec()
def test_dlogkdx_vec(dlogkdx_func, dlogkdx_vec_data, dtype):
    v, x, ans = dlogkdx_vec_data
    out = dlogkdx_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


logdkdv_func = gen_func_fixture("logdkdv", ["int"], ["jit"])


def test_logdkdv(logdkdv_func, logdkdv_data):
    v, x, ans = logdkdv_data
    out = logdkdv_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


@pytest.mark.vec()
def test_logdkdv_vec(logdkdv_func, logdkdv_vec_data):
    v, x, ans = logdkdv_vec_data
    out = logdkdv_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


logdkdx_func = gen_func_fixture("logdkdx", ["int"], ["jit"])


def test_logdkdx(logdkdx_func, logdkdx_data):
    v, x, ans = logdkdx_data
    out = logdkdx_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)


@pytest.mark.vec()
def test_logdkdx_vec(logdkdx_func, logdkdx_vec_data):
    v, x, ans = logdkdx_vec_data
    out = logdkdx_func(v, x)
    np.testing.assert_allclose(out, ans, rtol=tol)
