import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, random
from jax.scipy.special import erfinv


# CLF: V(x) = ||x - goal||^2
def vanilla_clf(state, goal):
    return jnp.linalg.norm(state - goal) ** 2

def vanilla_clf_x(state, goal):
    return ((state[0] - goal[0])**2).squeeze()

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
        self.alpha = alpha.reshape(-1, 1)
        self.beta = beta
        self.delta = delta
        self.n = n

    def extract_mu_sigma(self, b):
        mu = b[:self.n].reshape(-1, 1)  # Extract mean vector
        vec_sigma = b[self.n:]  # Extract upper triangular part of Sigma

        # Reconstruct full symmetric covariance matrix from vec(Sigma)
        sigma = jnp.zeros((self.n, self.n))
        upper_indices = jnp.triu_indices(self.n)  # Get upper triangular indices
        sigma = sigma.at[upper_indices].set(vec_sigma)
        sigma = sigma + sigma.T - jnp.diag(jnp.diag(sigma))  # Enforce symmetry

        return mu, sigma
    
    # @jit
    def get_b_vector(self, mu, sigma):

        # Extract the upper triangular elements of a matrix as a 1D array
        upper_triangular_indices = jnp.triu_indices(sigma.shape[0])
        vec_sigma = sigma[upper_triangular_indices]

        b = jnp.concatenate([mu, vec_sigma])

        return b
    
    def h_b(self, b):
        """Computes h_b(b) given belief state b = [mu, vec_u(Sigma)]"""
        mu, sigma = self.extract_mu_sigma(b)

        term1 = jnp.dot(self.alpha.T, mu) - self.beta
        term2 = jnp.sqrt(2 * jnp.dot(self.alpha.T, jnp.dot(sigma, self.alpha))) * erfinv(1 - 2 * self.delta)
        
        return (term1 - term2).squeeze()  # Convert from array to float
    
    def h_dot_b(self, b, dynamics):

        mu, sigma = self.extract_mu_sigma(b)
        
        # Compute gradient automatically - nx1 matrix containing partials of h_b wrt all n elements in b
        grad_h_b = jax.grad(self.h_b, argnums=0)

        def f_b(b):
            # Time update evaluated at mean
            f_vector = dynamics.f(b[:self.n]) 

            # Covariance update evaluated at mean
            A_f = jax.jacfwd(dynamics.f)(b[:self.n])
            f_sigma = A_f@sigma + sigma@(A_f.T) + dynamics.Q
            upper_triangular_indices = jnp.triu_indices(f_sigma.shape[0])
            f_sigma_vector = f_sigma[upper_triangular_indices]
            
            # "f" portion of belief vector
            f_b_vector = jnp.hstack([f_vector, f_sigma_vector])
            
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
                g_sigma = jnp.array([G_sigma[:, j][tri_indices] for j in range(m)]).T  # Shape (k, m)
                
                return g_sigma

            # Control influence on mean
            g_matrix = dynamics.g(b[:self.n])
            
            # Covariance update term
            A_g = jax.jacfwd(dynamics.g)(b[:self.n])
            # A_g = A_g.transpose(0, 2, 1) # nxnxm -> Remove if this is giving incorrect results (belief cbf was working for 2d dynamics before this was added for 4x1 dubins)
            g_sigma = A_g @ sigma + sigma @ (A_g.T) # No Q -> accounted for in f
            g_sigma_vector = extract_g_sigma(g_sigma)

            # "g" portion of belief vector
            g_b_matrix = jnp.vstack([g_matrix, g_sigma_vector])

            return g_b_matrix

        def L_f_h(b):
            return grad_h_b(b)@f_b(b)
        
        def L_g_h(b):
            return grad_h_b(b)@g_b(b)
        
        return L_f_h(b), L_g_h(b) 

        


# def get_diff(function, x, sigma):

#     subkey = SUBKEY # globally defined

#     n_samples = 10
#     state_dim = len(x)

#     samples = random.normal(subkey, (n_samples, state_dim))
#     jacobian = jacfwd(function)

#     total = 0
#     for sample in samples:
#         _, dyn_g = system_dynamics(sample[:-1])
#         grad = jacobian(sample)[:-1]
#         total += jnp.sum(jnp.abs(jnp.matmul(grad, dyn_g)))

#     def exponential_new_func(x: Array):
#         return jnp.matmul(jacobian(x)[:-1], system_dynamics(x[:-1])[0])


# def h_b_rb2(alpha, beta, mu, sigma, delta):
#     '''
#     Function for calculating barrier function for h_b where relative degree is 2
#     '''

#     roots = jnp.array([-0.1]) # Manually select root to be in left half plane
#     polynomial = np.poly1d(roots, r=True)
#     coeff = jnp.array(polynomial.coeffs)

#     h_0 = belief_cbf_half_space(alpha, beta, mu, sigma, delta)
#     h_1 = jnp.grad(h_0, argnums=2) # 1st order derivative, take derivative with respect to mean of belief

#     h_2 = jnp.array(h_0*coeff[0]) + jnp.array(h_1*coeff[1]) # equation 4 (belief paper), equation 38 (ECBF paper)

#     return h_2





    

