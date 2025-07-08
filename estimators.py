import os

import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import pandas as pd
from jax import jit
from jax.scipy.special import erf, erfinv
from openpyxl import load_workbook


class EKF:
    """Discrete EKF"""
    
    def __init__(self, dynamics, dt, x_init=None, P_init=None, Q=None, R=None):
        self.dynamics = dynamics  # System dynamics model
        self.dt = dt  # Time step
        self.K = jnp.zeros((dynamics.state_dim, dynamics.state_dim))
        self.name = "EKF"

        # Initialize belief (state estimate)
        self.x_hat = x_init if x_init is not None else jnp.zeros(dynamics.state_dim)

        # Covariance initialization
        self.P = P_init if P_init is not None else jnp.eye(dynamics.state_dim) * 0.1  
        self.Q = dynamics.Q # Process noise covariance
        self.R = R if R is not None else jnp.eye(dynamics.state_dim) * 0.05  # Measurement noise covariance

        self.in_cov = jnp.zeros((dynamics.state_dim, dynamics.state_dim)) # For tracking innovation covariance
        self.sigma_minus = self.P

    def predict(self, u):
        """Predict step of EKF."""
        # Nonlinear state propagation
        # self.x_hat = self.x_hat + self.dt * (self.dynamics.f(self.x_hat) + self.dynamics.g(self.x_hat) @ u)
        self.x_hat = self.x_hat + self.dt * self.dynamics.x_dot(self.x_hat, u)

        # Compute Jacobian of dynamics (linearization)
        # F_x = jax.jacobian(lambda x: x + self.dt * (self.dynamics.f(x) + self.dynamics.g(x) @ u))(self.x_hat)
        F = jax.jacobian(lambda x: (self.dynamics.x_dot(x, u).squeeze()))(self.x_hat)

        # Covariance udpate  (Page 274, Table 5.1, Optimal and Robust Estimation)
        # self.P = F_x @ self.P @ F_x.T + self.Q
        # P_dot = (F @ self.P + F.T @ self.P).squeeze() + self.Q
        P_dot = F.squeeze() @ self.P + self.P @ F.squeeze().T + self.Q
        self.P = self.P + P_dot*self.dt

    def update(self, z):
        """Measurement update step of EKF."""
        H_x = jnp.eye(self.dynamics.state_dim)  # Jacobian of measurement model (assuming direct state observation)
        y = z - self.x_hat # Innovation term: note self.x_hat comes from identity observation model

        # Innovation Covariance
        S = H_x @ self.P @ H_x.T + self.R

        # Handle degenerate S cases
        if jnp.linalg.norm(S) < 1e-8:
            S_inv = jnp.zeros_like(S)
        else:
            S_inv = jnp.linalg.pinv(S)

        self.K = self.P @ H_x.T @ S_inv

        # Update Innovation Covariance
        self.in_cov = self.K @ S @ self.K.T

        # Update state estimate
        self.x_hat = self.x_hat + y @ self.K

        self.sigma_minus = self.P # for computing probability bound

        # Update covariance
        self.P = (jnp.eye(max(z.shape)) - self.K @ H_x) @ self.P

    def get_belief(self):
        """Return the current belief (state estimate)."""
        return self.x_hat, self.P
    

    def compute_probability_bound(self, alpha, delta):
        """
        Returns the probability bounds for a range of delta values.
        """
        I = jnp.eye(self.K.shape[1])  # assuming K is (n x n)

        Sigma = self.sigma_minus
        K = self.K
        Lambda = self.in_cov
        H = jnp.eye(self.dynamics.state_dim) 
        
        alphaT_Sigma_alpha = alpha.T @ Sigma @ alpha
        term1 = jnp.sqrt(2 * alphaT_Sigma_alpha)
        term2 = jnp.sqrt(2 * alpha.T @ (I - K @ H) @ Sigma @ alpha)
        xi = erfinv(1 - 2 * delta) * (term1 - term2)

        denominator = jnp.sqrt(2 * alpha.T @ Lambda @ alpha)
        return 0.5 * (1 - erf(xi / denominator))


class GEKF:
    """Continuous-Discrete GEKF"""
    
    def __init__(self, dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, x_init=None, P_init=None, Q=None, R=None):
        self.dynamics = dynamics  # System dynamics model
        self.dt = dt  # Time step
        self.K = jnp.zeros((dynamics.state_dim, dynamics.state_dim))
        self.name = "GEKF"

        self.mu_u = mu_u
        self.sigma_u = sigma_u

        self.mu_v = mu_v
        self.sigma_v = sigma_v

        # Initialize belief (state estimate)
        self.x_hat = x_init if x_init is not None else jnp.zeros(dynamics.state_dim)

        # Covariance initialization
        self.P = P_init if P_init is not None else jnp.eye(dynamics.state_dim) * 0.1  
        self.Q = dynamics.Q # Process noise covariance
        self.R = sigma_v*jnp.eye(dynamics.state_dim) if R is not None else jnp.eye(dynamics.state_dim) * 0.05  # Measurement noise covariance

        self.in_cov = jnp.zeros((dynamics.state_dim, dynamics.state_dim)) # For tracking innovation covariance
        self.sigma_minus = self.P

    def predict(self, u):
        """Predict step of EKF."""
        # Nonlinear state propagation
        self.x_hat = self.x_hat + self.dt * self.dynamics.x_dot(self.x_hat, u)

        # Compute Jacobian of dynamics (linearization)
        F = jax.jacobian(lambda x: (self.dynamics.x_dot(x, u).squeeze()))(self.x_hat)

        # Covariance udpate  (Page 274, Table 5.1, Optimal and Robust Estimation)
        # P_dot = F @ self.P + self.P@ (F.T) + self.Q
        # P_dot = (F @ self.P + F.T @ self.P).squeeze() + self.Q
        P_dot = F.squeeze() @ self.P + self.P @ F.squeeze().T + self.Q

        self.P = self.P + P_dot*self.dt

    def update(self, z):
        """Measurement update step of EKF."""
        """z: measurement"""
        mu_u = self.mu_u
        sigma_u = self.sigma_u

        mu_v = self.mu_v
        sigma_v = self.sigma_v
      
        # Perfect state observation
        h_z = self.x_hat
        dhdx = jnp.eye(self.dynamics.state_dim) # change name of this variable

        h_z =  (1 + mu_u)*h_z
        E = h_z + mu_v

        C = (1 + mu_u)*jnp.matmul(self.P, jnp.transpose(dhdx, axes=None))  # Perform the matrix multiplication
        
        M = jnp.square(sigma_u)*jnp.diag(jnp.diag(jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx))) + jnp.matmul(h_z, jnp.transpose(h_z))))
        
        S = jnp.square(1 + mu_u)*jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx, axes=None)))  # Perform matrix multiplication
        S = S + M + jnp.square(sigma_v)

        self.K = jnp.matmul(C, jnp.linalg.inv(S))

        # Update state estimate
        self.x_hat = self.x_hat + jnp.matmul(z - E, self.K)

        # Update covariance
        self.P = self.P - jnp.matmul(self.K, jnp.transpose(C, axes=None))

    def update_1D(self, z):
        """Measurement update step of EKF."""
        """z: measurement"""

        mu_u = self.mu_u
        sigma_u = self.sigma_u

        mu_v = self.mu_v
        sigma_v = self.sigma_v
        
        # Perfect state observation
        h_z = self.x_hat
        dhdx = jnp.eye(self.dynamics.state_dim)

        h_z = (1+mu_u)*h_z
        E = h_z + mu_v

        C = (1+mu_u)*jnp.matmul(self.P, jnp.transpose(dhdx, axes=None))  # Perform the matrix multiplication
     
        M = jnp.diag(jnp.diag(jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx))) + jnp.matmul(h_z, jnp.transpose(h_z))))
        
        S = jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx, axes=None)))  # Perform matrix multiplication
        S = jnp.square(1 + mu_u)*S + jnp.square(sigma_u)*M + jnp.square(sigma_v)*jnp.eye(self.dynamics.state_dim)

        self.K = jnp.matmul(C, jnp.linalg.inv(S))

        # Update state estimate
        # self.x_hat = self.x_hat + jnp.matmul(self.K, z - E)
        self.x_hat = self.x_hat + jnp.matmul(z - E, self.K)

        # Update Innovation Covariance
        self.in_cov = self.K @ S @ self.K.T

        self.sigma_minus = self.P # for computing probability bound

        # Update covariance
        self.P = self.P - jnp.matmul(self.K, jnp.transpose(C, axes=None))

    def compute_probability_bound(self, alpha, delta):
        """
        Returns the probability bounds for a range of delta values.
        """
        I = jnp.eye(self.K.shape[1])  # assuming K is (n x n)

        Sigma = self.sigma_minus
        K = self.K
        Lambda = self.in_cov
        H = jnp.eye(self.dynamics.state_dim) 
        
        alphaT_Sigma_alpha = alpha.T @ Sigma @ alpha
        term1 = jnp.sqrt(2 * alphaT_Sigma_alpha)
        term2 = jnp.sqrt(2 * alpha.T @ (I - K @ H) @ Sigma @ alpha)
        xi = erfinv(1 - 2 * delta) * (term1 - term2)

        denominator = jnp.sqrt(2 * alpha.T @ Lambda @ alpha)
        return 0.5 * (1 - erf(xi / denominator))


    def get_belief(self):
        """Return the current belief (state estimate)."""
        return self.x_hat, self.P