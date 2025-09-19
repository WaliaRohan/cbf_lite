import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfinv
from jax.scipy.stats import norm
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
        if Q is None:
            self.Q = dynamics.Q # Process noise covariance
        else:
            self.Q = Q
        self.R = R if R is not None else jnp.eye(dynamics.state_dim) * 0.05  # Measurement noise covariance

        self.in_cov = jnp.zeros((dynamics.state_dim, dynamics.state_dim)) # For tracking innovation covariance
        self.innovation = jnp.array([[0.0]])

        self.sigma_minus = self.P
        self.mu_minus = self.x_hat

        # Initialize observation function as identity function
        if h is None:
            self.h = lambda x: x.ravel()
        else:
            self.h = h

        # Initialize S
        H_x = jax.jacfwd(self.h)(self.x_hat.ravel()) 
        obs_dim = len(H_x)
        self.S = H_x @ self.P @ H_x.T + self.R[:obs_dim, :obs_dim]

    @partial(jax.jit, static_argnums=0)   # treat `self` as static config
    def _predict(self, x_hat, P, u):
        f = self.dynamics.x_dot                   # pure function R^n×R^m→R^n (JAX ops only)
        x_next = x_hat + self.dt * f(x_hat, u)    # state propagation

        # Linearization wrt state (n×n)
        F = jax.jacfwd(f, argnums=0)(x_hat, u)

        # Covariance Euler step: Ṗ = F P + P Fᵀ + Q
        P_next = P + self.dt * (F @ P + P @ F.T + self.Q)
        P_next = 0.5*(P_next + P_next.T)
        return x_next, P_next

    def predict(self, u):
        """
        Same as predict step of EKF.
        
        See (Page 274, Table 5.1, Optimal and Robust Estimation)        
        """
        self.x_hat, self.P = self._predict(self.x_hat, self.P, u)


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
        self.S = S
        self.in_cov = self.K @ S @ self.K.T

        self.innovation = y

        # Update state estimate
        self.x_hat = self.x_hat + (self.K@y).reshape(self.x_hat.shape) # Order of K and y in multiplication matters!

        self.sigma_minus = self.P # For computing probability bound

        # Update covariance
        self.P = (jnp.eye(max(self.x_hat.shape)) - self.K @ H_x) @ self.P

        # # Joseph Stabilization
        # term = (jnp.eye(max(self.x_hat.shape)) - self.K @ H_x)

        # self.P = term @ self.P @ term.T + self.K @ self.R[:obs_dim, :obs_dim] @ (self.K.T)



    def get_belief(self):
        """Return the current belief (state estimate)."""
        return self.x_hat, self.P
    

    def prob_leaving_VaR_EKF(self, alpha, delta, beta):
        """
        Returns the probability bounds for a range of delta values.
        """

        def get_mult_std(alpha, cov):
            """
            Return std for multivariate random variable

            Args:
                alpha (vector): Linear gain
                cov (matrix): Covariance matrix

            Returns:
                float: std
            """
            return jnp.sqrt(2 * alpha.T @ cov @ alpha)

        # Compute xi b_minus
        I = jnp.eye(self.dynamics.state_dim)

        Sigma_minus = self.sigma_minus
        K = self.K
        Lambda = self.in_cov
        H = jax.jacfwd(self.h)(self.x_hat.ravel())
        var = (I - K@H)@Sigma_minus

        term1 = get_mult_std(alpha, Sigma_minus)
        term2 = get_mult_std(alpha, var)

        xi = erfinv(1 - 2 * delta) * (term1 - term2)

        denominator = get_mult_std(alpha, Lambda)

        prob = 0.5 * (1 - erf(xi / denominator))

        return prob

    def NEES(self, x):

        err = x - self.x_hat
        NEES = err.T @ jnp.linalg.inv(self.P) @ err

        return NEES

    def NIS(self):

        err = self.innovation
        NIS = err.T @ self.S @ err

        return NIS.squeeze()

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
        if Q is None:
            self.Q = dynamics.Q # Process noise covariance
        else:
            self.Q = Q
        self.R = R if R is not None else jnp.square(sigma_v)*jnp.eye(dynamics.state_dim) # Measurement noise covariance
         
        self.in_cov = jnp.zeros((dynamics.state_dim, dynamics.state_dim)) # For tracking innovation covariance
        self.innovation = jnp.array([[0.0]])

        # Initialize observation function as identity function
        if h is None:
            self.h = lambda x: x
        else:
            self.h = h

        # Pobability bound variables
        self.sigma_minus = self.P # Initialize covariance before measurement, for predicting probability bounds
        self.theta_prime = 0.0 # Initialize innovation term, for predicting probability bounds

        # H_x Jacobian of measurement function wrt state vector
        self.H_x = jax.jacfwd(self.h)(self.x_hat.ravel()) # The output of this should be (obs_dim, state_vector_len). 
        """
        NOTE: If H_x's shape is not (obs_dim, state_vector_len), ensure that the "h" operates on 1-dimensional
        state vector (x_dim, ) and the input (state vector value) at which jacobian needs to be calculted is also dimensionless.
        """

        self.obs_dim = len(self.H_x) # Number of rows in H_X == observation space dim

        # self.obs_dim = int(jnp.size(h(jnp.zeros(self.dynamics.state_dim))))
        self.K = jnp.zeros((self.dynamics.state_dim, self.obs_dim)) # Not sure if this matters. Other than for plotting. First Kalman gain get's updated during first measurement.

        # Initialize S
        H_x = jax.jacfwd(self.h)(self.x_hat.ravel()) 
        obs_dim = len(H_x)
        self.S = H_x @ self.P @ H_x.T + self.R[:obs_dim, :obs_dim]

    @partial(jax.jit, static_argnums=0)   # treat `self` as static config
    def _predict(self, x_hat, P, u):
        f = self.dynamics.x_dot                   # pure function R^n×R^m→R^n (JAX ops only)
        x_next = x_hat + self.dt * f(x_hat, u)    # state propagation

        # Linearization wrt state (n×n)
        F = jax.jacfwd(f, argnums=0)(x_hat, u)

        # Covariance Euler step: Ṗ = F P + P Fᵀ + Q
        P_next = P + self.dt * (F @ P + P @ F.T + self.Q)
        P_next = 0.5*(P_next + P_next.T)
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

        H_x = self.H_x
        obs_dim = self.obs_dim

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


        # Store info for plotting/analysis
        self.in_cov = S
        self.theta_prime = (self.K@y).reshape(self.x_hat.shape) # innovation term, double transpose because of how state is defined
        self.lambda_prime = self.K @ S @ self.K.T # innovation term covariance
        self.sigma_minus = self.P
        self.mu_minus = self.x_hat
        self.innovation = y
        self.S = S

        # Update state estimate
        self.x_hat = self.x_hat + self.theta_prime

        # Update covariance
        self.P = self.P - jnp.matmul(self.K, jnp.transpose(C))

    def prob_leaving_CVaR_GEKF(self, alpha, delta, beta):
        """
        Returns probability of leaving the safe set for the
        CVaR GEKF.

        Args:
            alpha (array): BCBF constraint state gain matrix
            beta (Float): BCBF constraint offset
            delta (float): desired risk level (probability of failure)
            h (function): BCBF function

        Returns:
            _type_: _description_
        """

        def get_mult_std(alpha, cov):
            """
            Return std for multivariate random variable

            Args:
                alpha (vector): Linear gain
                cov (matrix): Covariance matrix

            Returns:
                float: std
            """
            return jnp.sqrt(alpha.T @ cov @ alpha)

        # 1) Calculate ξ (xi) at b_minus (belief before measurement)
        I = jnp.eye(self.dynamics.state_dim)
        
        q_delta = norm.ppf(delta) # delta quantile of standard normal distribution
        f = norm.pdf(q_delta)
        
        var = (I - (1 + self.mu_u)*self.K@self.H_x)@self.sigma_minus

        xi_b_minus = f * (get_mult_std(alpha, self.sigma_minus) -  get_mult_std(alpha, var))/delta

        # 2) Calculate CVaR h_b at b_minus
        mu_mod = alpha.T @ self.mu_minus - beta
        sigma_mod = get_mult_std(alpha, self.sigma_minus)

        h_b_minus = mu_mod - (sigma_mod*f)/delta

        # 3) Return overall expression
        # num = alpha.T @ self.theta_prime + h_b_minus + xi_b_minus
        num = xi_b_minus

        prob_leaving = 0.5 * (1 - erf(num/(jnp.sqrt(2)*get_mult_std(alpha, self.lambda_prime))))

        return prob_leaving

    def prob_staying_CVaR_GEKF(self, alpha, delta, beta):

        return 1 - self.prob_leaving_CVaR_GEKF(alpha, delta, beta)

    # def NEES(self, x):
    #     """
    #     Calculate NEES at each timstep

    #     Args:
    #         x (jnp.array): True state vector

    #     Returns:
    #         Float: NEES value
    #     """

    #     err = x - self.x_hat
    #     NEES = err.T @ jnp.linalg.inv(self.P) @ err

    #     return NEES

    def NEES(self, x_true):
        err = x_true - self.x_hat              # (n,)
        P = 0.5 * (self.P + self.P.T)          # enforce symmetry

        # Solve using Cholesky instead of inverse
        L = jnp.linalg.cholesky(P)             # P = L L^T
        y = jnp.linalg.solve(L, err)           # L y = err
        nees_val = y @ y   

        return nees_val

    def NIS(self):

        err = self.innovation
        NIS = err.T @ jnp.linalg.inv(self.S) @ err

        return NIS.squeeze()

    def get_belief(self):
        """Return the current belief (state estimate)."""
        return self.x_hat, self.P