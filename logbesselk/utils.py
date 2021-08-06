import tensorflow as tf
import numpy as np


def get_deriv_func(func):
    def deriv(x):
        with tf.GradientTape() as g:
            g.watch(x)
            f = func(x)
        return g.gradient(f, x)
    return deriv


def extend(func, x, dx):

    def cond(x0, x1, f0, f1, cond):
        return tf.reduce_any(cond)

    def body(x0, x1, f0, f1, cond):
        x_extend = x0 + (x1 - x0) * 3
        f_extend = func(x_extend)
        x0_new = tf.where(cond, x1, x0)
        f0_new = tf.where(cond, f1, f0)
        x1_new = tf.where(cond, x_extend, x1)
        f1_new = tf.where(cond, f_extend, f1)
        cond = (x0_new != x1_new) & (f0_new * f1_new > 0)
        return x0_new, x1_new, f0_new, f1_new, cond

    x0, x1 = x, x + dx
    f0, f1 = func(x0), func(x1)
    x0, x1, f0, f1, _ = tf.while_loop(
        cond, body, (x0, x1, f0, f1, f0 * f1 > 0),
    )
    return x0, x1, f0, f1


def find_zero(func, x, dx, tol, max_iter=None):

    def cond(x0, x1, f0, f1, cond):
        return tf.reduce_any(cond)

    def body(x0, x1, f0, f1, cond):
        x_newton = x0 - f0 / deriv(x0)
        x_shrink = x0 + (x1 - x0) / 2

        xmin = tf.math.minimum(x0, x1)
        xmax = tf.math.maximum(x0, x1)
        x_newton = tf.clip_by_value(x_newton, xmin, xmax)

        f_newton = func(x_newton)
        f_shrink = func(x_shrink)

        select = tf.math.abs(f_newton / f_shrink) < 1
        x0_new = tf.where(select, x_newton, x_shrink)
        f0_new = tf.where(select, f_newton, f_shrink)
        x_other = tf.where(select, x_shrink, x_newton)
        f_other = tf.where(select, f_shrink, f_newton)

        x1_tmp = tf.where(f0 * f0_new <= 0, x0, x1)
        f1_tmp = tf.where(f0 * f0_new <= 0, f0, f1)
        x1_new = tf.where(f_other * f0_new <= 0, x_other, x1_tmp)
        f1_new = tf.where(f_other * f0_new <= 0, f_other, f1_tmp)
        return x0_new, x1_new, f0_new, f1_new, tf.abs(x0_new - x0) > tol

    deriv = get_deriv_func(func)
    x0, x1, f0, f1 = extend(func, x, dx)
    init = x0, x1, f0, f1, tf.ones_like(x0, tf.bool)
    return tf.while_loop(cond, body, init, maximum_iterations=max_iter)[0]


def find_zero_false_position(func, x, dx, tol, max_iter):

    def cond(x0, x1, f0, f1):
        return tf.reduce_any(tf.abs(x0 - x1) > tol)

    def body(x0, x1, f0, f1):
        x_new = x0 - f0 * (x1 - x0) / (f1 - f0)
        f_new = func(x_new)

        cond = f0 * f_new > 0
        x0_new = tf.where(cond, x_new, x0)
        f0_new = tf.where(cond, f_new, f0)
        x1_new = tf.where(cond, x1, x_new)
        f1_new = tf.where(cond, f1, f_new)
        return x0_new, x1_new, f0_new, f1_new

    init = extend(func, x, dx)
    return tf.while_loop(
        cond, body, init, maximum_iterations=max_iter,
    )[0]
