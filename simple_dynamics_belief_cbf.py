import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import vanilla_clf_x as clf
from dynamics import SimpleDynamics
from estimators import NonlinearEstimator as EKF
from sensor import noisy_sensor as sensor

# Define simulation parameters
dt = 0.01  # Time step
T = 1000 # Number of steps
u_max = 5.0

# Obstacle
wall_x = 2.0
goal_x = 10.0

# Initial state (truth)
x_true = jnp.array([0.0, 5.0])  # Start position
goal = jnp.array([goal_x, 5.0])  # Goal position
obstacle = jnp.array([wall_x, 0.0])  # Wall
safe_radius = 0.0  # Safety radius around the obstacle

dynamics = SimpleDynamics()
estimator = EKF(dynamics, sensor, dt, x_init=x_true)

# Define belief CBF parameters
n = 2
alpha = jnp.array([-1.0, 0.0])  # Example matrix
beta = jnp.array([-wall_x])  # Example vector
delta = 0.5  # Probability threshold
cbf = BeliefCBF(alpha, beta, delta, n)

# Control params
clf_gain = 0.1  # CLF linear gain
clf_slack = 10.0 # CLF slack
cbf_gain = 20.0  # CBF linear gain

# Autodiff: Compute Gradients for CLF and CBF
grad_V = grad(clf, argnums=0)  # ∇V(x)

# OSQP solver instance
solver = OSQP()

print(jax.default_backend())

@jit
def solve_qp(b):
    x_estimated, sigma = cbf.extract_mu_sigma(b)

    """Solve the CLF-CBF-QP using JAX & OSQP"""
    # Compute CLF components
    V = clf(x_estimated, goal)
    grad_V_x = grad_V(x_estimated, goal)  # ∇V(x)

    L_f_V = jnp.dot(grad_V_x.T, dynamics.f(x_estimated))
    L_g_V = jnp.dot(grad_V_x.T, dynamics.g(x_estimated))
    

    # Compute CBF components
    h_b = cbf.h_b(b)
    L_f_hb, L_g_hb = cbf.h_dot_b(b, dynamics)

    L_f_hb = L_f_hb.reshape(1, 1) # reshape to match L_f_V
    L_g_hb = L_g_hb.reshape(1, 2) # reshape to match L_g_V

    

    # Define QP matrices
    Q = jnp.eye(2)  # Minimize ||u||^2
    c = jnp.zeros(2)  # No linear cost term

    A = jnp.vstack([
        L_g_V,   # CLF constraint
        -L_g_hb,   # CBF constraint (negated for inequality direction)
        jnp.eye(2)
    ])

    u = jnp.hstack([
        (-L_f_V - clf_gain * V + clf_slack).squeeze(),   # CLF constraint
        (L_f_hb.squeeze() + cbf_gain * h_b).squeeze(),     # CBF constraint
        jnp.inf,
        u_max 
    ])

    l = jnp.hstack([
        -jnp.inf,
        -jnp.inf,
        -jnp.inf,
        -u_max
    ])

    # Solve the QP using jaxopt OSQP
    sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    return sol, V, h_b

x_traj = []  # Store trajectory
x_meas = [] # Measurements
x_est = [] # Estimates
u_traj = []  # Store controls
clf_values = []
cbf_values = []

x_traj.append(x_true)
x_estimated, p_estimated = estimator.get_belief()

def get_b_vector(mu, sigma):

    # Extract the upper triangular elements of a matrix as a 1D array
    upper_triangular_indices = jnp.triu_indices(sigma.shape[0])
    vec_sigma = sigma[upper_triangular_indices]

    b = jnp.concatenate([mu, vec_sigma])

    return b

# Simulation loop
for _ in tqdm(range(T), desc="Simulation Progress"):

    belief = get_b_vector(x_estimated, p_estimated)

    # Solve QP
    sol, V, h = solve_qp(belief)

    clf_values.append(V)
    cbf_values.append(h)

    u_opt = sol.primal[0]

    # Apply control to the true state (x_true)
    x_true = x_true + dt * (dynamics.f(x_true) + dynamics.g(x_true) @ u_opt)

    # obtain current measurement
    x_measured =  sensor(x_true)

    # updated estimate 
    estimator.predict(u_opt)
    estimator.update(x_measured)
    x_estimated, p_estimated = estimator.get_belief()

    # Store for plotting
    x_traj.append(x_true.copy())
    u_traj.append(u_opt)
    x_meas.append(x_measured)
    x_est.append(x_estimated)
    # print(x_true)

# Convert to JAX arrays
x_traj = jnp.array(x_traj)

# Conver to numpy arrays for plotting
x_traj = np.array(x_traj)
x_meas = np.array(x_meas)
x_est = np.array(x_est)

# Plot trajectory
plt.figure(figsize=(6, 6))
plt.plot(x_meas[:, 0], x_meas[:, 1], color="orange", linestyle=":", label="Measured Trajectory") # Plot measured trajectory
plt.plot(x_traj[:, 0], x_traj[:, 1], "b-", label="Trajectory (True state)")
plt.scatter(goal[0], goal[1], c="g", marker="*", s=200, label="Goal")

# Plot horizontal line at x = obstacle[0]
plt.axvline(x=obstacle[0], color="r", linestyle="--", label="Obstacle Boundary")

plt.xlabel("x")
plt.ylabel("y")
plt.title("CLF-CBF QP-Controlled Trajectory (Autodiff with JAX)")
plt.legend()
plt.grid()
plt.show()

# Second figure: X component comparison
plt.figure(figsize=(6, 4))
plt.plot(x_meas[:, 0], color="green", label="Measured x", linestyle="dashed")
plt.plot(x_est[:, 0], color="orange", label="Estimated x", linestyle="dotted")
plt.plot(x_traj[:, 0], color="blue", label="True x")
plt.xlabel("Time step")
plt.ylabel("X")
plt.legend()
plt.title("X Trajectory")
plt.show()

# Third figure: Y component comparison
plt.figure(figsize=(6, 4))
plt.plot(x_meas[:, 1], color="green", label="Measured y", linestyle="dashed")
plt.plot(x_est[:, 1], color="orange", label="Estimated y", linestyle="dotted")
plt.plot(x_traj[:, 1], color="blue", label="True y")
plt.xlabel("Time step")
plt.ylabel("Y")
plt.legend()
plt.title("Y Trajectory")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(cbf_values)
plt.xlabel("Time step")
plt.ylabel("CBF")
plt.title("CBF")
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(clf_values)
plt.xlabel("Time step")
plt.ylabel("CLF")
plt.title("CLF")
plt.show()

