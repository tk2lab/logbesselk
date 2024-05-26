# logbesselk
Provide function to calculate the modified Bessel function of the second kind
and its derivatives.

## Reference
Takashi Takekawa, Fast parallel calculation of modified Bessel function
of the second kind and its derivatives, SoftwareX, 17, 100923, 2022.

## Author
TAKEKAWA Takashi <takekawa@tk2lab.org>


## For Tensorflow

### Require
- Python (>=3.10)
- Tensorflow (>=2.8)

### Installation
```shell
pip install tensorflow logbesselk
```

### Examples
```python
import tensorflow as tf
from logbesselk.tensorflow import log_bessel_k as logk
from logbesselk.tensorflow import bessel_ke as ke
from logbesselk.tensorflow import bessel_kratio as kratio

v = 1.0
x = 1.0
a = logk(v, x)

v = tf.linspace(1, 10, 10)
x = tf.linspace(1, 10, 10)
b = logk(v, x)

# gradient
with tf.GradientTape() as g:
    g.watch(v, x)
    f = logk(v, x)
dlogkdv = g.gradient(f, v)
dlogkdx = g.gradient(f, x)

# use tf.function
logk = tf.function(logk)

# advanced version
from logbesselk.tensorflow import log_abs_deriv_bessel_k

logk = lambda v, x: log_abs_deriv_bessel_k(v, x, 0, 0)
logdkdv = lambda v, x: log_abs_deriv_bessel_k(v, x, 1, 0)
logdkdx = lambda v, x: log_abs_deriv_bessel_k(v, x, 0, 1)
```


## For jax

### Require
- Python (>=3.10)
- jax (>=0.4)

### Installation
```shell
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install logbesselk
```

### Examples
```python
import jax
import jax.numpy as jnp
from logbesselk.jax import log_bessel_k as logk
from logbesselk.jax import bessel_ke as ke
from logbesselk.jax import bessel_kratio as kratio

# scalar func and grad
v = 1.0
x = 1.0
a = logk(v, x)

# dlogK/dv = (dK/dv) / K
dlogkdv = jax.grad(logk, 0)
b = dlogkdv(v, x)

# dlogK/dx = (dK/dx) / K
dlogkdx = jax.grad(logk, 1)
c = dlogkdx(v, x)

# misc
d = ke(v, x)
e = kratio(v, x, d=1)

# vectorize
logk_vec = jax.vmap(logk)

v = jnp.linspace(1, 10, 10)
x = jnp.linspace(1, 10, 10)
f = logk_vec(v)

# use jit
logk_vec_jit = jax.jit(logk_vec)

# advanced version
from logbesselk.jax import log_abs_devel_bessel_k

log_dkdv = lambda v, x: log_abs_deriv_bessel_k(v, x, 1, 0)
log_dkdx = lambda v, x: log_abs_deriv_bessel_k(v, x, 0, 1)

log_dkdv_jit = jax.jit(jax.vmap(log_dkdv))
log_dkdx_jit = jax.jit(jax.vmap(log_dkdx))
```
