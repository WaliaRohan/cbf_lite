import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import norm
from functools import partial
from jax.scipy.special import erf, erfinv

@jax.jit
def sinusoidal_trajectory(t, A=1.0, omega=1.0, v=1.0, phase=0.0):
    """
    Generate a 2D sinusoidal trajectory.

    Parameters:
        t (array): Time vector
        state (array): State vector
        A (float): Amplitude of sine wave
        omega (float): Angular frequency of sine wave
        v (float): Constant forward velocity in x-direction
        phase (float): Phase shift of sine wave

    Returns:
        x (ndarray): x-positions (linear)
        y (ndarray): y-positions (sinusoidal)
    """
    # t = jnp.asarray(t)
    # x = v * t
    # y = A * jnp.sin(omega * t + phase)
    # return jnp.array([x, y])
    t = jnp.asarray(t)
    T = t[-1]
    
    # define scalar functions of time
    def x_of_t(tt):
        return v * tt

    def y_of_t(tt):
        amp_ramp = jnp.sin(0.5 * jnp.pi * tt / T)
        return (A * amp_ramp) * jnp.sin(omega * tt + phase)

    # vectorize values and time-derivatives
    x = jax.vmap(x_of_t)(t)
    y = jax.vmap(y_of_t)(t)
    x_dot = jax.vmap(jax.grad(x_of_t))(t)
    y_dot = jax.vmap(jax.grad(y_of_t))(t)

    theta = jnp.arctan2(y_dot, x_dot)
    return jnp.stack([x, y, theta], axis=0)

@jax.jit
def s_trajectory(T, A=1.0, omega=0.5, v=1.0):
    """
    Generates an S-shaped sinusoidal trajectory starting from (0,0).

    Args:
        T (jnp.ndarray): time indices, shape (N,)
        A (float): amplitude of sinusoid
        omega (float): frequency parameter
        v (float): forward velocity scale

    Returns:
        traj: jnp.ndarray of shape (3, N) containing x, y, theta
    """
    # nominal path
    x = v * T
    y = A * jnp.sin(omega * T) + A

    # shift to start at (0,0)
    x = x - x[0]
    y = y - y[0]

    # first derivatives
    dx = v * jnp.ones_like(T)                 # <-- FIX: positive
    dy = A * omega * jnp.cos(omega * T)

    # heading
    theta = jnp.arctan2(dy, dx)
    return jnp.stack([x, y, theta], axis=0)


@jax.jit
def straight_trajectory(T, y_val=0.0, lin_v=1.0):
    """
    Generates a straight trajectory along x-axis with constant y.

    Args:
        T (jnp.ndarray): time indices, shape (N,)
        y_val (float): constant y value
        lin_v (float): linear velocity in x direction

    Returns:
        x, y, theta: each shape (N,)
    """
    x = lin_v * T
    y = jnp.ones_like(T) * y_val
    theta = jnp.zeros_like(T)
    return jnp.stack([x, y, theta], axis=0)


@jax.jit
def gain_schedule_ctrl(v_r, x, x_d, ell=0.33, 
                             lambda1=1.0, kv=1.0, a1=1.0, a2=1.0):
    """
    Gain-scheduled controller (acceleration form).

    State x = [x, y, v, theta]
    Desired state x_d = [x_d, y_d, v_d, theta_d]

    Control output u = [a, omega]
      a     = longitudinal acceleration
      omega = yaw rate input
    """

    # Errors
    e_x = x[0] - x_d[0]
    e_y = x[1] - x_d[1]
    e_v = x[2] - v_r  # velocity error
    e_theta = x[3] - x_d[3]

    # Error vector
    e = jnp.array([e_x, e_v, e_y, e_theta])

    # Gains (scheduled)
    kx = lambda1
    ky = (a1 * ell) / (v_r**2)
    ktheta = (a2 * ell) / v_r

    # Gain matrix maps e -> [a, omega]
    K = jnp.array([
        [kx, kv, 0.0, 0.0],        # longitudinal channel
        [0.0, 0.0, ky, ktheta]     # lateral/heading channel
    ])

    # Nominal input (zero for straight-line ref)
    u_d = jnp.array([0.0, 0.0])

    # Control law
    u = u_d - K @ e

    return u

# @jax.jit
# def gain_schedule_ctrl(v_r, x, x_d, ell=0.05, lambda1=1.0, a1=1.0, a2=1.0):
#     """
#     Implements a gain scheduling based controller for trajectory tracking. It is
#     by linearizing dubin's dynamics about v = v_r, and theta = 0. This represents
#     the desired trajectory (x_d), which is essentially a straight path.

#     Source: R. Murray, Optimization-Based Control: Trajectory Generation and 
#             Tracking, v2.3h, Section 2.2

#     Args:
#         v_r (float): Desired longitudinal velocity magnitude.
#         x (array): State vector
#         x_d (array): Desired state vector (defined in source as [v_r*t, y_r, and theta])
#         ell (float, optional): wheelbase. Defaults to 0.33.
#         lambda1 (float, optional): closed loop eigen value of longitudinal dynamics (e_x). Defaults to 1.0.
#         a1 (float, optional): coeff 1 of polynomial equation for theta. Defaults to 2.0.
#         a2 (float, optional): coeff 2 of polynomial equation for theta. Defaults to 4.0.

#     Returns:
#         _type_: _description_
#     """
#     # Safe denominators for jit (avoid divide-by-zero near stops)
#     e = x - x_d
#     eps = 0.0 # 1e-6
#     vr = v_r
#     kx = lambda1
#     # ky = (a2 * ell) / (vr * vr + eps)
#     # ktheta = (a1 * ell) / (vr + eps)   # assumes vr>0 in normal use

#     # Tim Wheeler's formulation
#     ky = (a1 * ell)/jnp.square(v_r)
#     ktheta = (a2 * ell)/vr


#     K = jnp.array([[kx, 0.0, 0.0],
#                     [0, ky, ktheta]])

#     # w = u - u_d = [-kx*e_x, -(ky*e_y + ktheta*e_theta)]
#     # w1 = -kx * e[0]
#     # w2 = -(ky * e[1] + ktheta * e[2])

#     theta_d = x_d[-1]

#     rot = jnp.array([
#                     [jnp.cos(theta_d),  jnp.sin(theta_d), 0.0],
#                     [-jnp.sin(theta_d), jnp.cos(theta_d), 0.0],
#                     [              0.0,              0.0, 1.0]
#                     ])

#     x_ref = rot@x

#     # u = u_d + w with u_d = [v_r, 0]

#     u_d = jnp.array([vr, 0.0]) # Nominal steering angle. Currently zero?
#     u = u_d - K@e

#     # v = vr + w1
#     # delta = w2               
#     # return jnp.array([v, delta])

#     return u


def update_trajectory_index(system_pos, traj, index, eta):
    """
    Advance trajectory index if system is within eta of current target point.
    
    Inputs:
        system_pos (jnp.ndarray): shape (3,)
        traj (jnp.ndarray): shape (N, 2)
        index (int): current index
        eta (float): threshold distance

    Returns:
        new_index (int): updated index (may be unchanged)
    """
    current_target = traj[index]
    dist = jnp.linalg.norm(system_pos - current_target)
    return jnp.where(dist < eta, jnp.minimum(index + 1, traj.shape[0] - 1), index)

# CLF: V(x) = ||x - goal||^2
def vanilla_clf(state, goal):
    return jnp.linalg.norm(state - goal) ** 2

def vanilla_clf_dubins_2D(state, goal):

    x = state[0]
    y = state[1]
    theta = state[3]

    x_d = goal[0]
    y_d = goal[1]

    theta_d = jnp.atan2(y_d - y, x_d - x)

    theta_err = theta - theta_d

    alpha = 10.0
    beta = 10.0
    gamma = 1.0

    V = alpha*(x - x_d)**2 + beta*(y - y_d)**2 + gamma*theta_err**2 

    return V

def vanilla_clf_x(state, goal):
    return ((state[0] - goal[0])**2).squeeze()

# @jax.jit
def vanilla_clf_dubins(state, goal):
    state = jnp.asarray(state).reshape(-1)  # ensure 1-D
    goal  = jnp.asarray(goal).reshape(-1)

    y, v, theta = state[1], state[2], state[3]

    x_dot = v * jnp.cos(theta)
    y_dot = v * jnp.sin(theta)

    y_d = goal[1]
    e_y = (y_d - y) - y_dot

    lyap = 0.5 * e_y**2
    Kv = 0.5

    num = Kv * lyap + (y * y_dot)**2 + y_dot**2
    den = 1e-6 + (y * x_dot)**2

    return num / den

def clf_1D_doubleint(state, goal):
    """
    Returns lyapunov function for driving system to goal. This lyapunov function
    ensures LgV is not zero. Therefore, it can be used with a double integrator
    1D system. The (x - x_d)*\dot{x} term ensures that the value of this 
    function is zero at x = x_d. Both terms are squared to make sure that the
    function is positive definite. The \dot{x} term is responsible for ensuring
    that LgV is not zero.
 
    Args:
        state (numpy.ndarray): State vector [x x_dot].T
        goal (numpy.ndarray): 1D goal (desired x value)
    """
    x = state[0]
    x_dot = state[1]

    diff = x - goal

    # lyap = diff**2 + (x*x_dot)**2
    # lyap = diff**2 + diff*x_dot**2 
    # lyap = (diff - x_dot)**2 + x*x_dot**2 + x_dot**2
    # lyap = diff**2 + (diff**2)*x_dot**2
    # lyap = 0.5*(diff**2 + x_dot**2)
    # lyap = (diff + x_dot**2) + x_dot**2


    #  GAMMA: Controls settling time. 
    # LAMBDA: VERY SENSITIVE! Increasing this value helps reach goal faster. I think it also controls oscillations. This value should be low, around 0.5 - 1.0.
    #     MU: Helps in convergence to goal when system is near it. Near the goal, the "diff" (see below) might be close to zero. So this
              # relies on velocity to help the system reach the goal. 

    GAMMA = 2.4
    LAMBDA = 1.0
    MU = 1.0 
    lyap = (GAMMA*diff + LAMBDA*x_dot)**2 + MU*x_dot**2

    return lyap[0]
 

# CBF: h(x) = ||x - obstacle||^2 - safe_radius^2
def vanilla_cbf_circle(x, obstacle, safe_radius):
    return jnp.linalg.norm(x - obstacle) ** 2 - safe_radius**2

def vanilla_cbf_wall(state, obstacle):
    return obstacle[0] - state[0]

### Belief CBF

import numpy as np

KEY = random.PRNGKey(0)
KEY, SUBKEY = random.split(KEY)


class BeliefCBF:
    def __init__(self, alpha, beta, delta, n):
        """
        alpha: linear gain from half-space constraint (alpha^T.x >= B, where x is the state)
        beta: constant from half-space constraint
        delta: probability of failure (we want the system to have a probability of failure less than delta)
        n: dimension of the state space
        """
        self.alpha = alpha.reshape(-1, 1) # reshape into column vector
        self.beta = beta
        self.delta = delta
        self.n = n

    def extract_mu_sigma(self, b):
        mu = b[:self.n]  # Extract mean vector
        vec_sigma = b[self.n:] # Extract upper triangular part of Sigma

        # Reconstruct full symmetric covariance matrix from vec(Sigma)
        sigma = jnp.zeros((self.n, self.n))
        upper_indices = jnp.triu_indices(self.n)  # Get upper triangular indices
        sigma = sigma.at[upper_indices].set(vec_sigma)
        sigma = sigma + sigma.T - jnp.diag(jnp.diag(sigma))  # Enforce symmetry

        return mu, sigma
    
    @partial(jax.jit, static_argnums=0)   # treat `self` as static
    def get_b_vector(self, mu, sigma):

        # Extract the upper triangular elements of a matrix as a 1D array
        upper_triangular_indices = jnp.triu_indices(sigma.shape[0])
        vec_sigma = sigma[upper_triangular_indices]

        b = jnp.concatenate([mu.flatten(), vec_sigma]) # mu.squeeze() would not work for shapes of size (1, 1) (it deletes all 1 dimensions). mu.flatten() makes final shape (n, ), regardless of original shape. 

        return b

    def h_b(self, b):
        '''
        Computes CVaR belief CBF for a Multivariate Guassian Random Variable Y

        Reference: Calculating CVaR and bPOE for Common Probability Distributions With Application to Portfolio 
                   Optimization and Density Estimation  Matthew Norton · Valentyn Khokhlov · Stan Uryasev
        '''

        mu, sigma = self.extract_mu_sigma(b)

        mu_mod = jnp.dot(self.alpha.T, mu) - self.beta
        sigma_mod = jnp.sqrt(self.alpha.T @ sigma @ self.alpha) # sigma_mod is sd, whereas sigma is cov

        q_delta = norm.ppf(self.delta) # delta quantile of standard normal distribution

        f = norm.pdf(q_delta)

        CVaR = mu_mod - (sigma_mod*f)/self.delta # THIS IS FOR THE LOWER LEFT QUANTILE, DIFFERENT FROM bPOE SOURCE BUT CONSISTENT WITH BCBF

        return CVaR.squeeze() # Convert from array to float

    # def h_b(self, b):
    #     """Computes h_b(b) given belief state b = [mu, vec_u(Sigma)]"""
    #     mu, sigma = self.extract_mu_sigma(b)

    #     term1 = jnp.dot(self.alpha.T, mu) - self.beta
    #     term2 = jnp.sqrt(2 * jnp.dot(self.alpha.T, jnp.dot(sigma, self.alpha))) * erfinv(1 - 2 * self.delta)
        
    #     return (term1 - term2).squeeze()  # Convert from array to float
    # This entire function can be replaced by: term1 + jnp.sqrt(jnp.dot(self.alpha.T, jnp.dot(sigma, self.alpha))) * norm.ppf(self.delta))
    
    def h_dot_b(self, b, dynamics):

        mu, sigma = self.extract_mu_sigma(b)
        
        # Compute gradient automatically - nx1 matrix containing partials of h_b wrt all n elements in b
        grad_h_b = jax.grad(self.h_b, argnums=0)

        def extract_sigma_vector(sigma_matrix):
            """
            Extract f or g sigma by vectorizing the upper triangular part of each (n x n) slice in the given matrix.
            
            sigma_matrix: f_sigma or g_sigma
            """

            if sigma_matrix.shape == (1, 1):
                return sigma_matrix.flatten()

            else:
                shape = sigma_matrix.shape  # G_sigma is (n, m, n)
                n = shape[0]
                m = shape[1]

                # Create indices for the upper triangular part
                tri_indices = jnp.triu_indices(n)

                # Extract upper triangular elements from each m-th slice
                sigma_vector = jnp.array([sigma_matrix[:, j][tri_indices] for j in range(m)]).T # TODO: Check if this is valid for m > 1
                
                return sigma_vector

        def f_b(b):
            # Time update evaluated at mean
            f_vector = dynamics.f(b[:self.n]) 

            # Calculate A_f: This splitting of A into A_f is possible because sigma dot is control affine - see para after eq 43 in "Belief Space Planning ..." by Nishimura and Schwager
            A_f = jax.jacfwd(dynamics.f)(b[:self.n]) # jacfwd returns jacobian
            # Continuous time update - Optimal and Robust Control, Page 154, Eq 3.21
            f_sigma = A_f @ sigma + A_f.T @ sigma # Add noise (dynamics.Q) later -> Does it even matter if we're ultimately calculating dh/db (eq 19, BCBF paper)
            f_sigma_vector = extract_sigma_vector(f_sigma)
            
            # "f" portion of belief vector
            f_b_vector = jnp.vstack([f_vector, f_sigma_vector])
            
            return f_b_vector
        
        def g_b(b):

            def extract_g_sigma(G_sigma):
                """Extract g_sigma by vectorizing the upper triangular part of each (n x n) slice in G_sigma."""
                shape = G_sigma.shape  # G_sigma is (n, m, n)
                n = shape[0]
                m = shape[1]

                # Create indices for the upper triangular part
                tri_indices = jnp.triu_indices(n)

                # Extract upper triangular elements from each m-th slice
                g_sigma = jnp.array([G_sigma[:, j][tri_indices] for j in range(m)]).T # TODO: Check if this is valid for m > 1
                
                return g_sigma

            # Control influence on mean
            g_matrix = dynamics.g(b[:self.n])
            
            # Calculate A_g: This splitting of A into A_g is possible because sigma dot is control affine - see para after eq 43 in "Belief Space Planning ..." by Nishimura and Schwager
            A_g = jax.jacfwd(dynamics.g)(b[:self.n]) # jacfwd returns jacobian
            # A_g = A_g.transpose(0, 2, 1) # nxnxm -> Remove if this is giving incorrect results (belief cbf was working for 2d dynamics before this was added for 4x1 dubins)
            # Continuous time update - Optimal and Robust Control, Page 154, Eq 3.21
            g_sigma = A_g @ sigma + (A_g.T)@sigma  # No Q -> see "f_b" above
            g_sigma_vector = extract_g_sigma(g_sigma)

            # "g" portion of belief vector
            g_b_matrix = jnp.vstack([g_matrix, g_sigma_vector])

            return g_b_matrix

        def L_f_h(b):
            return jnp.reshape(grad_h_b(b) @ f_b(b), ())
        
        def L_g_h(b):
            return grad_h_b(b) @ g_b(b)
        
        def L_f_2_h(b):
            return jax.grad(L_f_h)(b) @ f_b(b)

        def Lg_Lf_h(b):
            
            return jax.grad(L_f_h)(b) @ g_b(b)

        return L_f_h(b), L_g_h(b), L_f_2_h(b), Lg_Lf_h(b), grad_h_b(b), f_b(b)
    
    def h_b_r2_RHS(self, h, L_f_h, L_f_2_h, cbf_gain):
        """
        Given a High-Order BCBF linear inequality constraint of relative
        degree 2: 

            h_ddot >= [alpha1 alpha2].T [h_dot h]

                where:
                    h_dot = LfH
                    h: position-based Belief Control Barrier Function constraint

        This function calculates the right-hand-side (RHS) of the following
        resulting QP linear inequality:

            -LgLfh * u <= -[alpha1 alpha2].T [Lfh h] + Lf^2h

        Args:
            b (jax.Array): belief vector
        
        Returns:
            float value: Value of RHS of the inequality above
        
        """        
        roots = jnp.array([-0.75]) # Manually select root to be in left half plane
        coeff = cbf_gain*jnp.poly(roots)

        # jax.debug.print("Value: {}", coeff)
        
        rhs = coeff@jnp.array([L_f_h, h]) + L_f_2_h

        return rhs, cbf_gain*coeff[0]*L_f_h, cbf_gain*coeff[1]*h

    

