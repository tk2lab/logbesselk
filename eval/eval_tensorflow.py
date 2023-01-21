import functools
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from logbesselk.tensorflow.integral import log_abs_deriv_bessel_k

logk = tf.function(log_abs_deriv_bessel_k)

dtype = tf.float64
eps = np.finfo(np.float64).eps
df = pd.read_csv("eval/data/logk_mathematica.csv")
v = tf.convert_to_tensor(df.v, dtype)
x = tf.convert_to_tensor(df.x, dtype)
res = logk(v, x)
df["res"] = res.numpy()
df["condition"] = np.square(df.v) / df.x
df["diff"] = df.res.astype(np.float64) - df.mathematica
df["rdiff"] = (df.res.astype(np.float64) - df.mathematica) / df.mathematica
df["absrdiff"] = np.abs(df.rdiff)
print("all case")
print(df[["diff", "rdiff", "absrdiff"]].describe())
df.sort_values("condition", inplace=True)
print("v^2/x < 100 case")
print(df.query("condition < 100")[["diff", "rdiff", "absrdiff"]].describe())

os.makedirs("eval/results")
df.to_csv("eval/results/logk_jax_int_64.csv")
