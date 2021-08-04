import tensorflow as tf


def get_deriv_func(func):
    def deriv(x):
        with tf.GradientTape() as g:
            g.watch(x)
            f = func(x)
        return g.gradient(f, x)
    return deriv


def find_zero(func, x, dx, tol, max_iter):

    def cond(x0, x1, f0, f1, diff):
        return tf.reduce_any(tf.abs(diff) > tol)

    def body(x0, x1, f0, f1, diff):
        x_newton = x0 - f0 / deriv(x0)
        x_shrink = x0 + (x1 - x0) / 2
        x_extend = x0 + (x1 - x0) * 3

        f_newton = func(x_newton)
        f_shrink = func(x_shrink)
        f_extend = func(x_extend)

        dn = (x_newton - x0) / (x1 - x0)
        shrink = tf.math.is_nan(x_newton) | (dn < 0) | (1 < dn)
        shrink |= (tf.math.abs(f_shrink / f_newton) < 1)
        x0_new = tf.where(shrink, x_shrink, x_newton)
        f0_new = tf.where(shrink, f_shrink, f_newton)
        x1_new = tf.where(f0 * f0_new < 0, x0, x1)
        f1_new = tf.where(f0 * f0_new < 0, f0, f1)
        
        x0_new = tf.where(f0 * f1 <= 0, x0_new, x1)
        f0_new = tf.where(f0 * f1 <= 0, f0_new, f1)
        x1_new = tf.where(f0 * f1 <= 0, x1_new, x_extend)
        f1_new = tf.where(f0 * f1 <= 0, f1_new, f_extend)
        
        x0_new = tf.where(x0 != x1, x0_new, x0)
        f0_new = tf.where(x0 != x1, f0_new, f0)
        x1_new = tf.where(x0 != x1, x1_new, x1)
        f1_new = tf.where(x0 != x1, f1_new, f1)
        return x0_new, x1_new, f0_new, f1_new, tf.math.abs(x0_new - x0)

    deriv = get_deriv_func(func)
    init = body(x, x + dx, func(x), func(x + dx), 10 * tol)
    return tf.while_loop(cond, body, init, maximum_iterations=max_iter)[0]
