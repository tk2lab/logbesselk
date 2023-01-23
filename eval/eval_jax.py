import functools
import os
os.environ["JAX_ENABLE_X64"] = "True"

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from logbesselk.jax.integral import log_abs_deriv_bessel_k


logk = jax.jit(jax.vmap(log_abs_deriv_bessel_k))

dtype = jnp.float64
eps = jnp.finfo(dtype).eps
df = pd.read_csv("eval/data/logk_mathematica.csv")
v = jnp.asarray(df.v, dtype)
x = jnp.asarray(df.x, dtype)
res = logk(v, x).block_until_ready()
df["res"] = np.asarray(res)
df["condition"] = np.square(df.v) / df.x
df["diff"] = df.res.astype(np.float64) - df.mathematica
df["rdiff"] = (df.res.astype(np.float64) - df.mathematica) / df.mathematica
df["absrdiff"] = np.abs(df.rdiff)
print("all case")
print(df[["diff", "rdiff", "absrdiff"]].describe())
df.sort_values("condition", inplace=True)
print("v^2/x < 100 case")
print(df.query("condition < 100")[["diff", "rdiff", "absrdiff"]].describe())

os.makedirs("eval/results", exist_ok=True)
df.to_csv("eval/results/logk_jax_int_64.csv")
