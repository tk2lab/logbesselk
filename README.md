# logbesselk

Provide function to calculate the modified Bessel function of the second kind.


## Examples

```python
import tensorflow as tf
from logbesselk.integral import log_bessel_k


n = 1000
v = 10. ** (2. * tf.random.uniform((n,), dtype=tf.float64) - 1.
x = 10. ** (3. * tf.random.uniform((n,), dtype=tf.float64) - 1.)

log_k = log_bessel_k(v, x)
log_dkdv = log_bessel_k(v, x, 1, 0)
log_dkdx = log_bessel_k(v, x, 0, 1)
```
