import numpy as np


def noisy_sensor(x_true):
    # Here we assume a simple noise-based estimator for demonstration.
    noise = np.random.normal(0, 0.1, size=x_true.shape)  # Adding Gaussian noise
    x_hat = x_true + noise  # Estimated state (belief)
    return x_hat


def identity_sensor(x_true):
    return x_true