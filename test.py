from functools import partial
import jax.numpy as jnp
from cbfs import belief_cbf_half_space as h_b
# from cbfs import b_dot
from dynamics import *

import jax
# Multi-dimensional
# mu = jnp.array([[0.5], [0.2], [-0.3], [0.1]])  # Example 4x1 mean vector
# sigma = jnp.eye(4) * 0.1  # Example 4x4 covariance matrix
# alpha = jnp.eye(4)  # Example 4x4 matrix
# beta = jnp.ones((4, 1))  # Example 4x1 vector
# delta = 0.05  # Probability threshold

# Single dimension
# mu = 0.5
# sigma = 0.1
# alpha = jnp.array([1.0])  # Example 4x4 matrix
# beta = 10.0 # Example 4x1 vector
# delta = 0.05  # Probability threshold

# # Partially apply h_b with fixed alpha, beta, and sigma
# h_b_partial = partial(h_b, alpha, beta, delta)

# result = h_b_partial(mu, sigma)
# print(result)

dynamics = SimpleDynamics()

dt = dt = 0.1
# Q = jnp.eye(len(dynamics.A)) * 0.01

mu = jnp.ones(2) * 0.5
sigma = jnp.eye(2) * 0.1

jac_f_x = jax.jacfwd(dynamics.f, argnums=0)(mu) # jacobian of f evaluated at mu
jac_g_x = jax.jacfwd(dynamics.g, argnums=0)(mu) # jacobian of g evaluated at mu

jac_f_x_sigma = jac_f_x@sigma + sigma@jac_f_x.T + dynamics.Q/2
jac_g_x_sigma = jac_g_x@sigma + sigma@jac_g_x.T + dynamics.Q/2



# mu_dot, sigma_dot = b_dot(mu, sigma, dynamics, Q, dt)

# beta = jnp.ones(len(dynamics.A))
# delta = 0.05