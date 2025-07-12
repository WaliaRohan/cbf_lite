import jax.numpy as jnp
import numpy as np
from jax import random


def identity_sensor(x_true):
    return x_true

# def noisy_sensor(x_true):
#     # Here we assume a simple noise-based estimator for demonstration.
#     noise = np.random.normal(0, 0.1, size=x_true.shape)  # Adding Gaussian noise
#     x_hat = x_true + noise  # Estimated state (belief)
#     return x_hat

def get_chol(cov, dim):
    """
    Returns the lower triangular matrix L for a positive definite covariance
    matrix generated from covariance cov: 

        Σ = L @ L.T, 

        where Σ is the covariance matrix,
        and L is the lower triangular matrix.

    Args:
        sigma (float): Covariance
        dim (int): Length of state vector

    Returns:
        chol (array): lower triangular matrix (L)
    """

    cov_matrix = cov * jnp.eye(dim)

    if jnp.trace(abs(cov_matrix)) > 0:
        chol = jnp.linalg.cholesky(cov_matrix)
    else:
        chol = jnp.zeros(cov_matrix.shape)

    return chol

def ubiased_noisy_sensor(x, t, std, key=None):
    """
    Applies additive zero-mean gaussian noise to true state value

    Args:
        x (array): true state
        t (int): simulation time step
        std (float): Std of additive noise
        key (int, optional): Key for random samples. Defaults to None.

    Returns:
        new_x (Array): Noisy measurement
    """

    if key is None:
        key = random.PRNGKey(0)

    key = random.fold_in(key, t) # create a new key for each time step, based on original key

    # Calculate the dimension of the random vector
    # dim = len(x)
    dim = max(x.shape)

    # (take n_initial_meas measurements at t = 0)
    n_initial_meas = 10
    max_iter = n_initial_meas if t == 0 else 1
    normal_samples = jnp.zeros((max_iter, dim))
    
    for ii in range(max_iter):
        key, subkey = random.split(key)

        # Populate with standard normal values
        normal_samples = normal_samples.at[ii, :].set(random.normal(subkey, shape=(dim,)))

    # Apply Cholesky decomposition to convert the unit variance vector to the desired covariance matrix
    # If Σ is a covariance matrix, then z=L⋅standard normal vector gives a sample with covariance Σ.

    chol_v = get_chol(std**2, dim)
    v_vector = jnp.mean(jnp.dot(chol_v, normal_samples.T), axis=1).reshape(x.shape)

    # new_x stores sensor measurement
    new_x = x
    new_x = new_x + v_vector # add biased gaussian noise

    return new_x

def noisy_sensor_mult(x, t, mu_u, sigma_u, mu_v, sigma_v, key=None):

    if key is None:
        key = random.PRNGKey(0)

    key = random.fold_in(key, t) # create a new key for each time step, based on original key

    # Calculate the dimension of the random vector
    # dim = len(x)
    dim = max(x.shape)

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

    # Apply Cholesky decomposition to convert the unit variance vector to the desired covariance matrix
    chol_u = get_chol(sigma_u**2, dim)
    chol_v = get_chol(sigma_v**2, dim)

    u_vector = 1 + mu_u + jnp.mean(jnp.dot(chol_u, normal_samples.T), axis=1)
    v_vector = mu_v + jnp.mean(jnp.dot(chol_v, normal_samples_2.T), axis=1)

    # new_x stores sensor measurement
    new_x = x

    # Add multiplicative noise to second state
    mult_state = 0
    if not jnp.isnan(jnp.mean(u_vector)):
        new_x = x.at[mult_state].set(x[mult_state]*jnp.mean(u_vector))

    new_x = new_x + v_vector # add biased gaussian noise

    return new_x
