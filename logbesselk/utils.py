import tensorflow as tf


def get_deriv_func(func):
    def deriv(t):
        with tf.GradientTape() as g:
            g.watch(t)
            d = func(t)
        return g.gradient(d, t)
    return deriv


def find_zero(func, x, dx, tol, max_iter):

    def cond(x0, x1, diff):
        return tf.reduce_any(tf.abs(diff) > tol)

    def body(x0, x1, diff):
        with tf.GradientTape() as g0:
            g0.watch(x0)
            f0 = func(x0)
        d0 = g0.gradient(f0, x0)
        with tf.GradientTape() as g1:
            g1.watch(x1)
            f1 = func(x1)
        d1 = g1.gradient(f1, x1)

        deriv = (f1 - f0) / (x1 - x0)
        newton = tf.math.is_finite(f0) & (tf.math.abs(deriv / d0) > 1)
        x_newton = tf.where(newton, x0 - f0 / d0, x1 - f1 / d1)
        f_newton = func(x_newton)

        x_shrink = x0 + (x1 - x0) / 2
        f_shrink = func(x_shrink)

        dn = (x_newton - x0) / (x1 - x0)
        shrink = (dn < 0) | (1 < dn) | (tf.math.abs(f_newton / f_shrink) > 1)
        x0_new = tf.where(shrink, x_shrink, x_newton)
        f0_new = tf.where(shrink, f_shrink, f_newton)
        x1_new = tf.where(f0_new * f0 > 0, x1, x0)
        
        extend = (f0 * f1 > 0.0)        
        x0_new = tf.where(extend, x1, x0_new)
        x1_new = tf.where(extend, x0 + (x1 - x0) * 3, x1_new)
        
        finish = x0 == x1
        x0_new = tf.where(finish, x0, x0_new)
        x1_new = tf.where(finish, x1, x1_new)
        return x0_new, x1_new, tf.math.abs(x0_new - x0)
    
    return tf.while_loop(
        cond, body, body(x, x + dx, 10 * tol), maximum_iterations=max_iter,
    )[0]
