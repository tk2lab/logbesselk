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
- Python (>=3.8)
- Tensorflow (>=2.6)

### Installation
```shell
pip install tensorflow logbesselk
```

### Examples
```python
import tensorflow as tf
from logbesselk.tensorflow import log_bessel_k

# return tensor
log_k = log_bessel_k(v=1.0, x=1.0)
log_dkdv = log_bessel_k(v=1.0, x=1.0, m=1, n=0)
log_dkdx = log_bessel_k(v=1.0, x=1.0, m=0, n=1)

# build graph at first execution time
log_bessel_k_tensor = tf.function(log_bessel_k)
log_bessel_dkdv_tensor = tf.function(lambda v, x: log_bessel_k(v, x, 1, 0))
log_bessel_dkdx_tensor = tf.function(lambda v, x: log_bessel_k(v, x, 0, 1))

n = 1000
for i in range(10):
    v = 10. ** (2. * tf.random.uniform((n,)) - 1.)
    x = 10. ** (3. * tf.random.uniform((n,)) - 1.)

    log_k = log_bessel_k_tensor(v, x)
    log_dkdv = log_bessel_dkdv_tensor(v, x)
    log_dkdx = log_bessel_dkdx_tensor(v, x)
```


## For jax

### Require
- Python (>=3.8)
- jax (>=0.3)

### Installation
```shell
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install logbesselk
```

### Examples
```python
import jax
from logbesselk.jax import log_bessel_k

# return jnp.array
log_k = log_bessel_k(v=1.0, x=1.0)
log_dkdv = log_bessel_k(v=1.0, x=1.0, m=1, n=0)
log_dkdx = log_bessel_k(v=1.0, x=1.0, m=0, n=1)

# build graph at first execution time
log_bessel_k_jit = jax.jit(jax.vmap(log_bessel_k))
log_bessel_dkdv_jit = jax.jit(jax.vmap(lambda v, x: log_bessel_k(v, x, 1, 0)))
log_bessel_dkdx_jit = jax.jit(jax.vmap(lambda v, x: log_bessel_k(v, x, 0, 1)))

trial = 10
n = 1000
for i in range(trial):
    v = 10. ** jax.random.uniform(i, (n,), minval=-1., maxval=1.)
    x = 10. ** jax.random.uniform(i, (n,), minval=-1., maxval=2.)

    log_k = log_bessel_k_tensor(v, x)
    log_dkdv = log_bessel_dkdv_tensor(v, x)
    log_dkdx = log_bessel_dkdx_tensor(v, x)
```
