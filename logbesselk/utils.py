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


def bessel_recurrence(ku0, ku1, u, x, n):

    def cond(k0, k1, ui, ni):
        return tf.reduce_any(ni != 0.)

    def body(k0, k1, ui, ni):
        kp = tk.log_add_exp(k0, k1 + tk.log(2. * (ui + 1.) / x))
        km = tk.log_sub_exp(k1, k0 + tk.log(2. * ui / x))
        k0 = tf.where(tf.equal(ni, 0.), k0, tf.where(ni > 0., k1, km))
        k1 = tf.where(tf.equal(ni, 0.), k1, tf.where(ni > 0., kp, k0))
        ui += tk.sign(ni)
        ni -= tk.sign(ni)
        return k0, k1, ui, ni

    init = tk.log(ku0), tk.log(ku1), u, n
    return tf.while_loop(cond, body, init)
