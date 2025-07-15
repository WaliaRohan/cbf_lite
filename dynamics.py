import jax.numpy as jnp

class SingleIntegrator1D:
    """1D Single Integrator Dynamics with Drift: dx/dt = a * x + u"""
    
    def __init__(self, a=0.1, b=1.0, Q=None):
        self.state_dim = 1
        self.name="Single Integrator 1D"
        self.a = a  # Drift coefficient
        self.f_matrix = jnp.array([a])  # Linear drift term
        self.g_matrix = jnp.array([[b]])  # Control directly influences state
        if Q is None:
            self.Q = jnp.eye(1) * 0  # Default to zero process noise
        else:
            self.Q = Q

    def f(self, x):
        """Drift dynamics: f(x)"""
        return self.f_matrix * x  # Linear drift

    def g(self, x):
        return self.g_matrix  # Constant control influence
    
    def x_dot(self, x, u):
        return self.f_matrix * x + self.g_matrix @ u
    

class NonLinearSingleIntegrator1D:
    """1D Single Integrator Dynamics with Drift: dx/dt = a * x + u"""
    
    def __init__(self, a=0.1, b=1.0, Q=None):
        self.state_dim = 1
        self.name="Single Integrator 1D"
        self.a = a  # Drift coefficient
        self.f_matrix = jnp.array([a])  # Linear drift term
        self.g_matrix = jnp.array([[b]])  # Control directly influences state
        if Q is None:
            self.Q = jnp.eye(1) * 0  # Default to zero process noise
        else:
            self.Q = Q

    def f(self, x):
        """Drift dynamics: f(x)"""
        return self.f_matrix * jnp.cos(x)  # Linear drift

    def g(self, x):
        return self.g_matrix  # Constant control influence
    
    def x_dot(self, x, u):
        return self.f(x)+ self.g(x) @ u

class SimpleDynamics:
    """Simple system dynamics: dx/dt = f(x) + g(x) u"""
    
    def __init__(self, Q=None):
        self.state_dim = 2
        self.f_matrix = jnp.array([[0.01, 0.02], [0.03, 0.04]])  # No drift for now
        self.g_matrix = jnp.array([[1, 0], [0, 1]])  # Identity control matrix
        if Q is None:
            self.Q = jnp.eye(self.f_matrix.shape[1])*0 
        else:
            self.Q = Q

    def f(self, x):
        """Drift dynamics: f(x)"""
        return self.f_matrix @ x  # Linear drift (zero in this case)

    def g(self, x):
        """Control matrix: g(x)"""
        return self.g_matrix  # Constant control input mapping
    
    def x_dot(self, x, u):
        return self.f_matrix@x + self.g_matrix@u

class LinearDoubleIntegrator1D:
    """1D Linear Double Integrator:
    State: x = [x1 (position), x2 (velocity)]
    Dynamics: dx1/dt = x2, dx2/dt = u
    """

    def __init__(self, Q=None):
        self.state_dim = 2
        self.name = "Linear Double Integrator 1D"

        # Control influence matrix: u affects acceleration (x2_dot)
        self.g_matrix = jnp.array([[0.0],
                                   [1.0]])

        # Default process noise
        if Q is None:
            self.Q = jnp.eye(2) * 0
        else:
            self.Q = Q

    def f(self, x):
        """
        x: nx1 column vector ([x, v].T)
        """
        x_flat = x.ravel() # for jit compatibility
        x2 = x_flat[1]
        return jnp.array([[x2], [0.0]])  # dx1 = x2, dx2 = 0

    def g(self, x):
        """
        Control influence: constant
        Note: argument x is not required. However, this function signature is
        required to be compatible with the simulation code.
        """
        return self.g_matrix

    def x_dot(self, x, u):
        """Total dynamics: dx/dt = f(x) + g(x)u"""

        # Reshape u to ensure it has atleast 1 column - for compatibility with sim code
        return (self.f(x) + self.g(x) @ (u.reshape(u.shape[0], -1))).reshape(x.shape)
    
class NonlinearSingleIntegrator:
    """Nonlinear single integrator dynamics: dx/dt = f(x) + g(x) u"""
    
    def __init__(self, Q=None):
        self.state_dim = 2
        if Q is None:
            self.Q = jnp.eye(2) * 0
        else:
            self.Q = Q
    
    def f(self, x):
        """Nonlinear drift dynamics: f(x)"""
        return jnp.array([
            jnp.sin(x[0]).squeeze(),
            jnp.cos(x[1]).squeeze()
        ])
    
    def g(self, x):
        """State-dependent control matrix: g(x)"""
        return jnp.array([
            [1 + 0.1 * jnp.sin(x[0]).squeeze(), 0],
            [0, 1 + 0.1 * jnp.cos(x[1]).squeeze()]
        ])
    
    def x_dot(self, x, u):
        return self.f(x) + self.g(x) @ u

class DubinsDynamics:
    """2D Dubins Car Model with constant velocity and control over heading rate."""

    def __init__(self, Q=None):
        self.name = "Dubins Dynamics"
        self.state_dim = 4
        """Initialize Dubins Car dynamics."""
        if Q is None:
            self.Q = jnp.eye(4)*0 
        else:
            self.Q = Q

    def f(self, x):
        """
        Compute the drift dynamics f(x).
        
        State x = [x_pos, y_pos, v, theta]
        """
        v = x[2]
        theta = x[3]
        
        return jnp.array([
            [v * jnp.cos(theta)],  # x_dot
            [v * jnp.sin(theta)],  # y_dot
            [jnp.zeros_like(v)],   # v_dot (velocity is constant)
            [jnp.zeros_like(theta)]    # theta_dot (no drift)
        ])

    def g(self, x):
        """
        Compute the control matrix g(x).
        
        Control u = [heading rate omega]
        """
        return jnp.array([
            [0],  # No control influence on x
            [0],  # No control influence on y
            [0],  # No control influence on velocity
            [1]   # Control directly affects theta (heading)
        ])

    def x_dot(self, x, u):
        """Total dynamics: dx/dt = f(x) + g(x)u"""

        # Reshape u to ensure it has atleast 1 column
        return (self.f(x) + self.g(x) @ (u.reshape(u.shape[0], -1))).reshape(x.shape)
    
