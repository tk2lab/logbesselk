import pytest
import pandas as pd
import numpy as np

from logbesselk.integral import log_bessel_k
from logbesselk.integral import bessel_ke
from logbesselk.integral import bessel_k_ratio


def test_logk(datadir):
    df = pd.read_csv('./data/logk_mathematica.csv')
    v = df['v']
    x = df['x']
    val = df['true']
    assert np.all(np.isclose(log_bessel_k(v, x).numpy(), val))


def test_logdkdv(datadir):
    df = pd.read_csv('./data/log_dkdv_mathematica.csv')
    v = df['v']
    x = df['x']
    val = df['true']
    assert np.all(np.isclose(log_bessel_k(v, x, 1, 0).numpy(), val))


def test_logdkdx(datadir):
    df = pd.read_csv('./data/log_dkdx_mathematica.csv')
    v = df['v']
    x = df['x']
    val = df['true']
    assert np.all(np.isclose(log_bessel_k(v, x, 0, 1).numpy(), val))


def test_ke(datadir):
    df = pd.read_csv('./data/ke_mathematica.csv')
    v = df['v']
    x = df['x']
    val = df['true']
    assert np.all(np.isclose(bessel_ke(v, x).numpy(), val))


def test_ke(datadir):
    df = pd.read_csv('./data/kratio_mathematica.csv')
    v = df['v']
    x = df['x']
    val = np.array(df['true'])
    calc = bessel_k_ratio(v, x).numpy()
    assert np.all(np.isclose(calc, val))
