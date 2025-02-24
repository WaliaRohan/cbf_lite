import jax
import jax.numpy as jnp
from cbfs import BeliefCBF
from dynamics import SimpleDynamics

# Multi-dimensional
n = 2
alpha = jnp.array([1.0, 0.0])  # Example matrix
beta = jnp.array([5.0])  # Example vector
delta = 0.05  # Probability threshold

cbf = BeliefCBF(alpha, beta, delta, n)

mu = jnp.array([0.2, -0.3])
vec_sigma = jnp.array([0.1, 0.05, 0.2])  # Upper triangular part
b = jnp.concatenate([mu, vec_sigma])

output = cbf.h_b(b)

print (output)

# Compute gradient automatically
grad_h_b = jax.grad(cbf.h_b, argnums=0)(b)

print(grad_h_b(b))
