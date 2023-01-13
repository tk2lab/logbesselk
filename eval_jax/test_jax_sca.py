import time

import jax
import jax.numpy as jnp
import jax.lax as lax

from logbesselk.jax.sca import log_bessel_k


log_bessel_k_jit = jax.jit(jax.vmap(jax.vmap(log_bessel_k)))
log_bessel_k_dx_jit = jax.jit(jax.vmap(jax.vmap(jax.grad(log_bessel_k, 1))))

v, x = jnp.meshgrid(jnp.linspace(0, 99, 1000), jnp.linspace(0, 99, 1000))

log_bessel_k_jit(v, x).block_until_ready()
start = time.time()
out = log_bessel_k_jit(v, x).block_until_ready()
print(time.time() - start)
print(out)

log_bessel_k_dx_jit(v, x).block_until_ready()
start = time.time()
out = log_bessel_k_dx_jit(v, x).block_until_ready()
print(time.time() - start)
print(out)
