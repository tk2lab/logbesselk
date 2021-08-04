import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from logbesselk import log_bessel_k, log_bessel_dkdv, log_bessel_dkdx


if __name__ == '__main__':

    for dtype in [np.float32]:
        print(f'dtype: {dtype}')
        for v in 10 ** np.arange(-20, 21, 5, dtype=dtype):
            print(f'v: {v}')
            v = tf.constant(v)
            x = tf.constant(10 ** np.linspace(-20.0, 20.0, 6, dtype=dtype))
            y = tfp.math.log_bessel_kve(v, x) - x
            z = log_bessel_k(v, x)
            tf.print(v)
            tf.print(tf.stack([x, y, z], axis=-1))
