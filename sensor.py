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

    You can then generate random samples: x = mu + L @ z
    
    Here z: uncorrelated normal samples ~ N(0, I). Identity covariance means independent components
    
    Covariance of Lz becomes Σ. Proof:
    
    E[yy^T] = E[Lz (Lz)^T] 
             = E[L z z^T L^T] 
             = L E[zz^T] L^T 
             = L * I * L^T       # (since E[zz^T] = I for standard normal z)
             = L L^T             # (covariance of y)

    Args:
        sigma (float): Covariance
        dim (int): Length of state vector

    Returns:
        L (array): lower triangular matrix (L)
    """

    cov_matrix = cov * jnp.eye(dim)

    if jnp.trace(abs(cov_matrix)) > 0:
        L = jnp.linalg.cholesky(cov_matrix)
    else:
        L = jnp.zeros(cov_matrix.shape)

    return L

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
    dim = max(x.shape)

    # Generate standard normal samples (zero-mean Gaussian random vector with unit variance)
    # (take n_initial_meas measurements at t = 0)
    n_initial_meas = 10
    max_iter = n_initial_meas if t == 0 else 1
    normal_samples = jnp.zeros((max_iter, dim))
    
    for ii in range(max_iter):
        key, subkey = random.split(key)
        normal_samples = normal_samples.at[ii, :].set(random.normal(subkey, shape=(dim,)))

    # Apply Cholesky decomposition to convert the unit variance vector to the desired covariance matrix
    L_v = get_chol(std**2, dim)
    v_vector = jnp.mean(jnp.dot(L_v, normal_samples.T), axis=1).reshape(x.shape)

    # new_x stores sensor measurement
    new_x = x
    new_x = new_x + v_vector # add biased gaussian noise

    return new_x

def noisy_sensor_mult(x, t, mu_u, sigma_u, mu_v, sigma_v, key=None):

    if key is None:
        key = random.PRNGKey(0)

    key = random.fold_in(key, t) # create a new key for each time step, based on original key

    # Calculate the dimension of the random vector
    dim = max(x.shape)

    # Generate standard normal samples (zero-mean Gaussian random vector with unit variance)
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

    # Apply Cholesky decomposition to obtain lower triangular matrix L of covariance Σ
    L_u = get_chol(sigma_u**2, dim)
    L_v = get_chol(sigma_v**2, dim)

    u_vector = mu_u + jnp.mean(jnp.dot(L_u, normal_samples.T), axis=1)
    v_vector = mu_v + jnp.mean(jnp.dot(L_v, normal_samples_2.T), axis=1)

    # new_x stores sensor measurement
    new_x = x

    # Add multiplicative noise and biased guassian noise
    new_x = (1 + u_vector)*new_x + v_vector

    return new_x
