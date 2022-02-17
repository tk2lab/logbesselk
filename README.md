# logbesselk
Provide function to calculate the modified Bessel function of the second kind
and its derivatives.


## Author
TAKEKAWA Takashi <takekawa@tk2lab.org>


# Reference
Takashi Takekawa, Fast parallel calculation of modified Bessel function
of the second kind and its derivatives, SoftwareX, 17, 100923, 2022.


### Require
- python >= 3.7.1
- tensorflow >= 2.6.1


## Installation
```shell
pip install logbesselk
```


## Examples
```python
import tensorflow as tf
from logbesselk.integral import log_bessel_k


log_k = log_bessel_k(v=1.0, x=1.0)
log_dkdv = log_bessel_k(v=1.0, x=1.0, 1, 0)
log_dkdx = log_bessel_k(v=1.0, x=1.0, 0, 1)


# build graph at first execution time
log_bessel_k_tensor = tf.function(log_bessel_k)
log_bessel_dkdv_tensor = tf.function(lambda v, x: log_bessel_k(v, x, 1, 0))
log_bessel_dkdx_tensor = tf.function(lambda v, x: log_bessel_k(v, x, 0, 1))

n = 1000
for i in range(10):
    v = 10. ** (2. * tf.random.uniform((n,), dtype=tf.float64) - 1.
    x = 10. ** (3. * tf.random.uniform((n,), dtype=tf.float64) - 1.)

    log_k = log_bessel_k_tensor(v, x)
    log_dkdv = log_bessel_dkdv_tensor(v, x)
    log_dkdx = log_bessel_dkdx_tensor(v, x)
```


## Evaluation
```shell
python -m eval.prec
python -m eval.time
python -m eval.scale
python -m eval.fig1
python -m eval.fig2
python -m eval.fig3
python -m eval.fig4
python -m eval.fig5
python -m eval.fig6
python -m eval.fig7
```
