import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfinv
from functools import partial

class EKF:
    """Discrete EKF"""
    
    def __init__(self, dynamics, dt, h = None, x_init=None, P_init=None, Q=None, R=None):
        self.dynamics = dynamics  # System dynamics model
        self.dt = dt  # Time step
        self.K = jnp.zeros((dynamics.state_dim, dynamics.state_dim)) # Ideally, it's shape should be (dynamics.state_dim, obs_dim)
        self.name = "EKF"

        # Initialize belief (state estimate)
        self.x_hat = x_init if x_init is not None else jnp.zeros(dynamics.state_dim)

        # Covariance initialization
        self.P = P_init if P_init is not None else jnp.eye(dynamics.state_dim) * 0.1  
        self.Q = dynamics.Q # Process noise covariance
        self.R = R if R is not None else jnp.eye(dynamics.state_dim) * 0.05  # Measurement noise covariance

        self.in_cov = jnp.zeros((dynamics.state_dim, dynamics.state_dim)) # For tracking innovation covariance
        self.sigma_minus = self.P

        # Initialize observation function as identity function
        if h is None:
            self.h = lambda x: x.ravel()
        else:
            self.h = h

    def predict(self, u):
        """
        Predict step of EKF.
        
        See (Page 274, Table 5.1, Optimal and Robust Estimation)
        """
        # Nonlinear state propagation
        self.x_hat = self.x_hat + self.dt * self.dynamics.x_dot(self.x_hat, u)

        # Compute Jacobian of dynamics (linearization)
        F = jax.jacobian(lambda x: (self.dynamics.x_dot(x, u).squeeze()))(self.x_hat)

        # Covariance udpate 
        if len(F.shape) > 2: 
            F = F.squeeze()
        P_dot = F @ self.P + self.P @ F.T + self.Q
        self.P = self.P + P_dot*self.dt

    def update(self, z):
        """
        Measurement update step of EKF

        Args:
            z (): Measurement
        """
        # H_x Jacobian of measurement function wrt state vector
        H_x = jax.jacfwd(self.h)(self.x_hat.ravel()) # The output of this should be (obs_dim, state_vector_len). 
        """
        NOTE: If H_x's shape is not (obs_dim, state_vector_len), ensure that the "h" operates on 1-dimensional
        state vector (x_dim, ) and the input (state vector value) at which jacobian needs to be calculted is also dimensionless.
        """
        
        obs_dim = len(H_x) # Number of rows in H_X == observation space dim

        z_obs = self.h(z) # This might not be technically correct, but here I am just extracting the second state from the measurement

        y = (z_obs - self.h(self.x_hat)) # Innovation term: note self.x_hat comes from identity observation model
        y = jnp.reshape(y, (obs_dim, 1)) 

        # Innovation Covariance
        S = H_x @ self.P @ H_x.T + self.R[:obs_dim, :obs_dim]# self.

        # Handle degenerate S cases
        if jnp.linalg.norm(S) < 1e-8:
            S_inv = jnp.zeros_like(S)
        else:
            S_inv = jnp.linalg.pinv(S)

        self.K = self.P @ H_x.T @ S_inv

        # Update Innovation Covariance (For calculating probability bound)
        self.in_cov = self.K @ S @ self.K.T

        # Update state estimate
        self.x_hat = self.x_hat + (self.K@y).reshape(self.x_hat.shape) # Order of K and y in multiplication matters!

        self.sigma_minus = self.P # For computing probability bound

        # Update covariance
        self.P = (jnp.eye(max(self.x_hat.shape)) - self.K @ H_x) @ self.P

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
    
    def __init__(self, dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, h = None, x_init=None, P_init=None, Q=None, R=None):
        self.dynamics = dynamics  # System dynamics model
        self.dt = dt  # Time step
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
        self.R = R if R is not None else jnp.square(sigma_v)*jnp.eye(dynamics.state_dim) # Measurement noise covariance
         
        self.in_cov = jnp.zeros((dynamics.state_dim, dynamics.state_dim)) # For tracking innovation covariance
        self.sigma_minus = self.P # For computing probability bound

        # Initialize observation function as identity function
        if h is None:
            self.h = lambda x: x
        else:
            self.h = h

        self.obs_dim = int(jnp.size(h(jnp.zeros(self.dynamics.state_dim))))
        self.K = jnp.zeros((self.dynamics.state_dim, self.obs_dim)) # Not sure if this matters. Other than for plotting. First Kalman gain get's updated during first measurement.


    @partial(jax.jit, static_argnums=0)   # treat `self` as static config
    def _predict(self, x_hat, P, u):
        f = self.dynamics.x_dot                   # pure function R^n×R^m→R^n (JAX ops only)
        x_next = x_hat + self.dt * f(x_hat, u)    # state propagation

        # Linearization wrt state (n×n)
        F = jax.jacfwd(f, argnums=0)(x_hat, u)

        # Covariance Euler step: Ṗ = F P + P Fᵀ + Q
        P_next = P + self.dt * (F @ P + P @ F.T + self.Q)
        return x_next, P_next

    def predict(self, u):
        """
        Same as predict step of EKF.
        
        See (Page 274, Table 5.1, Optimal and Robust Estimation)        
        """
        self.x_hat, self.P = self._predict(self.x_hat, self.P, u)

    def update(self, z):
        """
        Measurement update step of GEKF.
        z: measurement
        """
        mu_u = self.mu_u
        sigma_u = self.sigma_u
        mu_v = self.mu_v

        # H_x Jacobian of measurement function wrt state vector
        H_x = jax.jacfwd(self.h)(self.x_hat.ravel()) # The output of this should be (obs_dim, state_vector_len). 
        """
        NOTE: If H_x's shape is not (obs_dim, state_vector_len), ensure that the "h" operates on 1-dimensional
        state vector (x_dim, ) and the input (state vector value) at which jacobian needs to be calculted is also dimensionless.
        """

        obs_dim = len(H_x) # Number of rows in H_X == observation space dim

        z_obs = self.h(z) # This might not be technically correct, but here I am just extracting the second state from the measurement

        h_z = self.h(self.x_hat)
        E = (1 + mu_u)*h_z + mu_v # This is the "observation function output" for GEKF

        y = (z_obs - E) # Innovation term: note self.x_hat comes from identity observation model
        y = jnp.reshape(y, (obs_dim, 1)) 

        dhdx = H_x

        C = (1 + mu_u)*jnp.matmul(self.P, jnp.transpose(dhdx))  # Perform the matrix multiplication
        
        M = jnp.diag(jnp.diag(jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx))) + jnp.matmul(h_z, jnp.transpose(h_z))))
        
        S_term_1 = jnp.square(1 + mu_u)*jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx)))  # Perform matrix multiplication
        S = S_term_1 + jnp.square(sigma_u)*M + self.R[:obs_dim, :obs_dim]

        self.K = jnp.matmul(C, jnp.linalg.inv(S))

        # Update state estimate
        self.x_hat = self.x_hat + (self.K@y).reshape(self.x_hat.shape) # double transpose because of how state is defined.

        # Update covariance
        self.P = self.P - jnp.matmul(self.K, jnp.transpose(C))

    # @partial(jax.jit, static_argnums=0)   # treat `self` as static
    # def _update(self, x_hat, P, z):
    #     mu_u = self.mu_u
    #     sigma_u = self.sigma_u
    #     mu_v = self.mu_v
    #     obs_dim = self.obs_dim

    #      # H_x Jacobian of measurement function wrt state vector
    #     H_x = jax.jacfwd(self.h)(self.x_hat.ravel()) # The output of this should be (obs_dim, state_vector_len). 
    #     """
    #     NOTE: If H_x's shape is not (obs_dim, state_vector_len), ensure that the "h" operates on 1-dimensional
    #     state vector (x_dim, ) and the input (state vector value) at which jacobian needs to be calculted is also dimensionless.
    #     """

    #     obs_dim = len(H_x) # Number of rows in H_X == observation space dim

    #     z_obs = self.h(z) # This might not be technically correct, but here I am just extracting the second state from the measurement

    #     h_z = self.h(self.x_hat)
    #     E = (1 + mu_u)*h_z + mu_v # This is the "observation function output" for GEKF

    #     y = (z_obs - E) # Innovation term: note self.x_hat comes from identity observation model
    #     y = jnp.reshape(y, (obs_dim, 1)) 

    #     dhdx = H_x

    #     C = (1 + mu_u)*jnp.matmul(self.P, jnp.transpose(dhdx))  # Perform the matrix multiplication
        
    #     M = jnp.diag(jnp.diag(jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx))) + jnp.matmul(h_z, jnp.transpose(h_z))))
        
    #     S_term_1 = jnp.square(1 + mu_u)*jnp.matmul(dhdx, jnp.matmul(self.P, jnp.transpose(dhdx)))  # Perform matrix multiplication
    #     S = S_term_1 + jnp.square(sigma_u)*M + self.R[:obs_dim, :obs_dim]

    #     # K = jnp.matmul(C, jnp.linalg.inv(S))
    #     K = jnp.linalg.solve(S, C.T).T  

    #     # Update state estimate
    #     x_hat_next = self.x_hat + (self.K@y).reshape(self.x_hat.shape) # double transpose because of how state is defined.

    #     # Update covariance
    #     P_next = self.P - jnp.matmul(self.K, jnp.transpose(C))

    #     return x_hat_next, P_next, K

    # def update(self, z):
    #     self.x_hat, self.P, self.K = self._update(self.x_hat, self.P, z)

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