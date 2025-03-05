import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg


class EKF:
    """Discrete EKF"""
    
    def __init__(self, dynamics, sensor_model, dt, x_init=None, P_init=None, Q=None, R=None):
        self.dynamics = dynamics  # System dynamics model
        self.sensor_model = sensor_model  # Sensor model
        self.dt = dt  # Time step

        # Initialize belief (state estimate)
        self.x_hat = x_init if x_init is not None else jnp.zeros(2)

        # Covariance initialization
        self.P = P_init if P_init is not None else jnp.eye(2) * 0.1  
        self.Q = dynamics.Q # Process noise covariance
        self.R = R if R is not None else jnp.eye(2) * 0.05  # Measurement noise covariance

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
        H_x = jnp.eye(2)  # Jacobian of measurement model (assuming direct state observation)
        # y = z - self.sensor_model(self.x_hat)  # Innovation (difference between measured and predicted state)
        y = z - self.x_hat # self.x_hat combes from identity observation model

        # Kalman gain
        S = H_x @ self.P @ H_x.T + self.R
        K = self.P @ H_x.T @ linalg.inv(S)

        # Update state estimate
        self.x_hat = self.x_hat + K @ y

        # Update covariance
        self.P = (jnp.eye(2) - K @ H_x) @ self.P

    def get_belief(self):
        """Return the current belief (state estimate)."""
        return self.x_hat, self.P


class GEKF:
    """Continuous-Discrete GEKF"""
    
    def __init__(self, dynamics, sensor_model, dt, x_init=None, P_init=None, Q=None, R=None):
        self.dynamics = dynamics  # System dynamics model
        self.sensor_model = sensor_model  # Sensor model
        self.dt = dt  # Time step

        # Initialize belief (state estimate)
        self.x_hat = x_init if x_init is not None else jnp.zeros(2)

        # Covariance initialization
        self.P = P_init if P_init is not None else jnp.eye(2) * 0.1  
        self.Q = dynamics.Q # Process noise covariance
        self.R = R if R is not None else jnp.eye(2) * 0.05  # Measurement noise covariance

    def predict(self, u):
        """Predict step of EKF."""
        # Nonlinear state propagation
        self.x_hat = self.x_hat + self.dt * (self.dynamics.f(self.x_hat) + self.dynamics.g(self.x_hat) @ u)

        # Jacobian of noise-free model evaluate at current mean (self.x_hat)
        A_f = jax.jacobian(self.dynamics.f)(self.x_hat)
        A_g = jax.jacobian(self.dynamics.g)(self.x_hat)
        F = A_f + jnp.einsum("ijk,j->ik", A_g, u)

        # # Compute Jacobian of dynamics (linearization)
        # F_x = jax.jacobian(lambda x: x + self.dt * (self.dynamics.f(x) + self.dynamics.g(x) @ u))(self.x_hat)

        # Continous covariance udpate
        P_dot = F @ self.P + self.P@ F + self.Q
        self.P = self.P + P_dot*self.dt

    def update(self, z, t):
        """Measurement update step of EKF."""
        """z: measurement"""
        
        # Multiplicative noise
        mu_u = 0.0174
        sigma_u = 10*2.916e-4 # 10 times more than what was shown in GEKF paper

        # Additive noise
        mu_v = -0.0386
        sigma_v = 7.997e-5
        
        # h_z = self.sensor_model(self.x_hat, t) # replace with h_z = h(z) for more complex models
        # H_dot_fn = jax.jacfwd(self.sensor_model, argnums=0)  # Jacobian of measurement model (assuming direct state observation), Differentiate w.r.t x
        # H_dot = H_dot_fn(self.x_hat, t)
      
        # Perfect state observation
        h_z = z
        H_dot = jnp.eye(len(z))

        h_z = h_z.at[1].set(h_z[1] * (1 + mu_u))
        E = h_z + mu_v

        C = jnp.matmul(self.P, jnp.transpose(H_dot, axes=None))  # Perform the matrix multiplication
        C = C.at[1].set((1 + mu_u) * C[1])  # Modify only the second element
        
        M = jnp.diag(jnp.diag(jnp.matmul(H_dot, jnp.matmul(self.P, jnp.transpose(H_dot))) + jnp.matmul(h_z, jnp.transpose(h_z))))
        
        S = jnp.matmul(H_dot, jnp.matmul(self.P, jnp.transpose(H_dot, axes=None)))  # Perform matrix multiplication
        S = S.at[1].set(jnp.square(1 + mu_u) * S[1])  # Apply (1 + mu_u)^2 to the second element only
        M = M.at[1, 1].set(jnp.square(sigma_u) * M[1, 1])
        S = S + M + jnp.square(sigma_v)

        K = jnp.matmul(C, jnp.linalg.inv(S))

        # Update state estimate
        self.x_hat = self.x_hat + jnp.matmul(K, z - E)

        # Update covariance
        self.P = self.P - jnp.matmul(K, jnp.transpose(C, axes=None))

    def get_belief(self):
        """Return the current belief (state estimate)."""
        return self.x_hat, self.P