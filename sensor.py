import jax.numpy as jnp
import numpy as np
from jax import random


def identity_sensor(x_true):
    return x_true

def noisy_sensor(x_true):
    # Here we assume a simple noise-based estimator for demonstration.
    noise = np.random.normal(0, 0.1, size=x_true.shape)  # Adding Gaussian noise
    x_hat = x_true + noise  # Estimated state (belief)
    return x_hat

def get_chol(sigma, dim):

    cov_matrix = sigma * jnp.eye(dim)

    # Apply Cholesky decomposition to convert the unit variance vector to the desired covariance matrix
    if jnp.trace(abs(cov_matrix)) > 0:
        chol = jnp.linalg.cholesky(cov_matrix)
    else:
        chol = jnp.zeros(cov_matrix.shape)

    return chol

def noisy_sensor_mult(x, t, key=None):
    # Multiplicative noise
    mu_u = 0.0174
    sigma_u = 10*2.916e-4 # 10 times more than what was shown in GEKF paper

    # Additive noise
    mu_v = -0.0386
    sigma_v = 7.997e-5

    if key is None:
        key = random.PRNGKey(0)

    key = random.fold_in(key, t) # create a new key for each time step, based on original key

    # Calculate the dimension of the random vector
    dim = len(x)

    # Generate a zero-mean Gaussian random vector with unit variance
    # (take n_initial_meas measurements at t = 0)
    n_initial_meas = 10
    max_iter = n_initial_meas if t == 0 else 1
    normal_samples = jnp.zeros((max_iter, dim))
    normal_samples_2 = jnp.zeros((max_iter, dim))
    
    for ii in range(max_iter):
        key, subkey = random.split(key)
        key, subkey2 = random.split(key)
        normal_samples = normal_samples.at[ii, :].set(random.normal(subkey, shape=(dim,)))
        normal_samples_2 = normal_samples_2.at[ii, :].set(random.normal(subkey2, shape=(dim,)))


    chol_u = get_chol(sigma_u, dim)
    chol_v = get_chol(sigma_v, dim)

    u_vector = 1 + mu_u + jnp.mean(jnp.dot(chol_u, normal_samples.T), axis=1)
    v_vector = mu_v + jnp.mean(jnp.dot(chol_v, normal_samples.T), axis=1)

    new_x = x + v_vector # add biased gaussian noise

    # Add multiplicative noise to second state
    state_idx = 1
    if not jnp.isnan(jnp.mean(u_vector)):
        new_x = x.at[state_idx].set(x[state_idx]*jnp.mean(u_vector))

    return new_x
