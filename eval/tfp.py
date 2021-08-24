import tensorflow_probability as tfp


def log_bessel_k(v, x):
    return tfp.math.log_bessel_kve(v, x) - x
