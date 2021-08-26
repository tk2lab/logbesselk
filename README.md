# logbesselk

Provide function to calculate the modified Bessel function of the second kind.


## Examples

```
import tensorflow as tf
from logbesselk.integrate import log_bessel_k


n = 1000
v = 100. * tf.random.uniform((n,))
x = 10. ** (3. * tf.random.uniform((n,)) - 1.)

log_bessel_k(v, x)
```
