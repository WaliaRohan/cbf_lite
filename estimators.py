import jax
import jax.numpy as jnp
import jax.scipy.linalg as linalg


class NonlinearEstimator:
    """Extended Kalman Filter (EKF) for state estimation with internal belief tracking."""
    
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

        # Update covariance
        self.P = F_x @ self.P @ F_x.T + self.Q

    def update(self, z):
        """Measurement update step of EKF."""
        H_x = jnp.eye(2)  # Jacobian of measurement model (assuming direct state observation)
        y = z - self.sensor_model(self.x_hat)  # Innovation (difference between measured and predicted state)

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
