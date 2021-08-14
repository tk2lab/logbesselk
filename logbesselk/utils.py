import tensorflow as tf

from . import math as tk


def get_deriv_func(func):
    def deriv(x):
        with tf.GradientTape() as g:
            g.watch(x)
            f = func(x)
        return g.gradient(f, x)
    return deriv


def extend(func, x, dx):

    def cond(x0, x1, f0, f1):
        return tf.reduce_any((x0 != x1) & (f1 > 0))

    def body(x0, x1, f0, f1):
        x0_new = tf.where(f1 > 0, x1, x0)
        f0_new = tf.where(f1 > 0, f1, f0)
        x1_new = tf.where(f1 > 0, x0 + 3 * (x1 - x0), x1)
        f1_new = tf.where(f1 > 0, func(x1_new), f1)
        return x0_new, x1_new, f0_new, f1_new

    x0, x1 = x, x + dx
    f0 = func(x0)
    f1 = func(x1)
    init = x0, x1, f0, f1
    x0, x1, f0, f1 = tf.while_loop(cond, body, init)
    return x1, x0, f1, x0


def find_zero(func, x, dx, n_iter):

    def cond(x0, x1, f0, f1):
        return True

    def body(x0, x1, f0, f1):
        x_shrink = x0 + 0.5 * (x1 - x0)
        f_shrink = func(x_shrink)

        dx = - f0 / deriv(x0) / (x1 - x0)
        dx_in_range = (0.0 < dx) & (dx < 0.5)
        x_newton = tf.where(dx_in_range, x0 + dx * (x1 - x0), x0)
        f_newton = tf.where(dx_in_range, func(x_newton), f0)

        c_shrink = f_shrink < 0
        c_newton = f_newton < 0
        x0_new = tf.where(c_shrink, x_shrink, tf.where(c_newton, x_newton, x0))
        f0_new = tf.where(c_shrink, f_shrink, tf.where(c_newton, f_newton, f0))
        x1_new = tf.where(c_shrink, x1, tf.where(c_newton, x_shrink, x_newton))
        f1_new = tf.where(c_shrink, f1, tf.where(c_newton, f_shrink, f_newton))
        return x0_new, x1_new, f0_new, f1_new

    deriv = get_deriv_func(func)
    init = extend(func, x, dx)
    x0, x1, f0, f1 = tf.while_loop(cond, body, init, maximum_iterations=n_iter)
    return x0


def log_bessel_recurrence(log_ku, log_kup1, u, n, x):

    def cond(ki, kj, ui, ni):
        return tf.reduce_any(ni != 0.)

    def body(ki, kj, ui, ni):
        uj = ui + tk.sign(ni)
        nj = ni - tk.sign(ni)
        kp = tk.log_add_exp(ki, kj + tk.log(2. * uj / x))
        km = tk.log_sub_exp(kj, ki + tk.log(2. * ui / x))
        kj = tf.where(tf.equal(ni, 0.), ki, tf.where(ni > 0, kj, km))
        kk = tf.where(tf.equal(ni, 0.), kj, tf.where(ni > 0, kp, ki))
        return kj, kk, uj, nj

    init = log_ku, log_kup1, u, n
    return tf.while_loop(cond, body, init)[:2]


def wrap_K(log_k, log_dk_dv, log_minus_dk_dx, v, x, name=None):

    @tf.custom_gradient
    def custom_gradient(v, x):
        def grad(u):
            if log_minus_dk_dx is None
                dkvdx = (v / x) * kv - tk.exp(log_k(v + 1., x))
            else:
                dkvdx = -tk.exp(log_minus_dk_dx(v, x))
            if log_dk_dv is None:
                return None, u * dkdvx
            else:
                dkvdv = tk.exp(log_dk_dv(v, x))
                return u * dkvdv, u * dkvdx
        kv = tk.exp(log_k(v, x))
        return kv, grad

    with tf.name_scope(name or 'bessel_K'):
        x = tf.convert_to_tensor(x)
        v = tf.convert_to_tensor(v, x.dtype)
        return gradient(v, x)


def wrap_log_K(log_k, log_dk_dv, log_minus_dk_dx, v, x, name=None):

    @tf.custom_gradient
    def custom_gradient(v, x):
        def grad(u):
            if log_minsu_dk_dx is None:
                dlogkvdx = v / x - tk.exp(log_k(v + 1, x) - logkv)
            else:
                dlogkvdx = -tk.exp(log_minus_dk_dx(v, x) - logkv)
            if log_dk_dv is None:
                return None, u * dlogkvdx
            else:
                dlogkvdv = tk.exp(log_dk_dv(v, x) - logkv)
                return u * dlogkvdv, u * dlogkvdx
        logkv = log_k(v, x)
        return logkv, grad

    with tf.name_scope(name or 'log_K'):
        x = tf.convert_to_tensor(x)
        v = tf.convert_to_tensor(v, x.dtype)
        return custom_gradient(v, x)
