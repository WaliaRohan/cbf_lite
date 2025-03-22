import os

import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg
import pandas as pd
from jax import jit
from openpyxl import load_workbook


class EKF:
    """Discrete EKF"""
    
    def __init__(self, dynamics, sensor_model, dt, x_init=None, P_init=None, Q=None, R=None):
        self.dynamics = dynamics  # System dynamics model
        self.sensor_model = sensor_model  # Sensor model
        self.dt = dt  # Time step
        self.K = jnp.zeros((dynamics.state_dim, dynamics.state_dim))
        self.name = "EKF"

        # Initialize belief (state estimate)
        self.x_hat = x_init if x_init is not None else jnp.zeros(dynamics.state_dim)

        # Covariance initialization
        self.P = P_init if P_init is not None else jnp.eye(dynamics.state_dim) * 0.1  
        self.Q = dynamics.Q # Process noise covariance
        self.R = R if R is not None else jnp.eye(dynamics.state_dim) * 0.05  # Measurement noise covariance

    def predict(self, u):
        """Predict step of EKF."""
        # Nonlinear state propagation
        self.x_hat = self.x_hat + self.dt * (self.dynamics.f(self.x_hat) + self.dynamics.g(self.x_hat) @ u)

        # Compute Jacobian of dynamics (linearization)
        F_x = jax.jacobian(lambda x: x + self.dt * (self.dynamics.f(x) + self.dynamics.g(x) @ u))(self.x_hat)

        # Discrete covariance update
        self.P = F_x @ self.P @ F_x.T + self.Q

    def update(self, z):
        """Measurement update step of EKF."""
        H_x = jnp.eye(len(self.x_hat))  # Jacobian of measurement model (assuming direct state observation)
        # y = z - self.sensor_model(self.x_hat)  # Innovation (difference between measured and predicted state)
        y = z - self.x_hat # Innovation term: note self.x_hat comes from identity observation model

        # Kalman gain
        S = H_x @ self.P @ H_x.T + self.R
        self.K = self.P @ H_x.T @ linalg.inv(S)

        # Update state estimate
        self.x_hat = self.x_hat + self.K @ y

        # Update covariance
        self.P = (jnp.eye(len(z)) - self.K @ H_x) @ self.P

    def get_belief(self):
        """Return the current belief (state estimate)."""
        return self.x_hat, self.P

class GEKF:
    """Continuous-Discrete GEKF"""
    
    def __init__(self, dynamics, sensor_model, dt, x_init=None, P_init=None, Q=None, R=None):
        self.dynamics = dynamics  # System dynamics model
        self.sensor_model = sensor_model  # Sensor model
        self.dt = dt  # Time step
        self.K = jnp.zeros((dynamics.state_dim, dynamics.state_dim))
        self.name = "GEKF"

        # Initialize belief (state estimate)
        self.x_hat = x_init if x_init is not None else jnp.zeros(dynamics.state_dim)

        # Covariance initialization
        self.P = P_init if P_init is not None else jnp.eye(dynamics.state_dim) * 0.1  
        self.Q = dynamics.Q # Process noise covariance
        self.R = R if R is not None else jnp.eye(dynamics.state_dim) * 0.05  # Measurement noise covariance

    def predict(self, u):
        """Predict step of EKF."""
        # Nonlinear state propagation
        self.x_hat = self.x_hat + self.dt * (self.dynamics.f(self.x_hat) + self.dynamics.g(self.x_hat) @ u)

        # Compute Jacobian of dynamics (linearization)
        F = jax.jacobian(lambda x: self.dynamics.f(x) + self.dynamics.g(x) @ u)(self.x_hat)

        # Continous covariance udpate
        P_dot = F @ self.P + self.P@ (F.T) + self.Q
        self.P = self.P + P_dot*self.dt

    def update(self, z):
        """Measurement update step of EKF."""
        """z: measurement"""

        mult_state = 0
        
        # Multiplicative noise
        mu_u = 0.0174
        sigma_u = jnp.sqrt(2.916e-4) # 10 times more than what was shown in GEKF paper

        # Additive noise
        mu_v = -0.0386
        sigma_v = jnp.sqrt(7.97e-5)
        
        # h_z = self.sensor_model(self.x_hat, t) # replace with h_z = h(z) for more complex models
        # dhdx_fn = jax.jacfwd(self.sensor_model, argnums=0)  # Jacobian of measurement model (assuming direct state observation), Differentiate w.r.t x
        # dhdx = dhdx_fn(self.x_hat, t)
      
        # Perfect state observation
        h_z = self.x_hat
        dhdx = jnp.eye(len(self.x_hat)) # change name of this variable

        h_z = h_z.at[mult_state].set(h_z[mult_state] * (1 + mu_u))
        E = h_z + mu_v

        C = jnp.matmul(self.P, jnp.transpose(dhdx, axes=None))  # Perform the matrix multiplication
        C = C.at[mult_state].set((1 + mu_u) * C[mult_state])  # Modify only the specified state
        
        M = jnp.diag(jnp.diag(jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx))) + jnp.matmul(h_z, jnp.transpose(h_z))))
        
        S = jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx, axes=None)))  # Perform matrix multiplication
        S = S.at[mult_state].set(jnp.square(1 + mu_u) * S[mult_state])  # Apply (1 + mu_u)^2 to the specified state
        M = M.at[1, 1].set(jnp.square(sigma_u) * M[1, 1])
        S = S + M + jnp.square(sigma_v)

        self.K = jnp.matmul(C, jnp.linalg.inv(S))

        # Update state estimate
        self.x_hat = self.x_hat + jnp.matmul(self.K, z - E)

        # Update covariance
        self.P = self.P - jnp.matmul(self.K, jnp.transpose(C, axes=None))

    def update_1D(self, z):
        """Measurement update step of EKF."""
        """z: measurement"""

        mult_state = 0
        
        # mu_u = 0.0174
        # sigma_u = jnp.sqrt(2.916e-4) # 10 times more than what was shown in GEKF paper

        # # Additive noise
        # mu_v = -0.0386
        # sigma_v = jnp.sqrt(7.97e-5)

        mu_u = 0.1
        sigma_u = jnp.sqrt(0.001)

        mu_v = 0.01
        sigma_v = jnp.sqrt(0.0001)
        
        # Perfect state observation
        h_z = self.x_hat
        dhdx = jnp.eye(len(self.x_hat))

        h_z = (1+mu_u)*h_z
        E = h_z + mu_v

        C = (1+mu_u)*jnp.matmul(self.P, jnp.transpose(dhdx, axes=None))  # Perform the matrix multiplication
     
        M = jnp.diag(jnp.diag(jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx))) + jnp.matmul(h_z, jnp.transpose(h_z))))
        
        S = jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx, axes=None)))  # Perform matrix multiplication
        S = jnp.square(1 + mu_u)*S + jnp.square(sigma_u)*M + jnp.square(sigma_v)*jnp.eye(self.dynamics.state_dim)

        self.K = jnp.matmul(C, jnp.linalg.inv(S))

        # Update state estimate
        self.x_hat = self.x_hat + jnp.matmul(self.K, z - E)

        # Update covariance
        self.P = self.P - jnp.matmul(self.K, jnp.transpose(C, axes=None))


    def get_belief(self):
        """Return the current belief (state estimate)."""
        return self.x_hat, self.P