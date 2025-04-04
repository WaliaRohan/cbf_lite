import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import vanilla_clf_x as clf
from dynamics import DubinsDynamics, NonlinearSingleIntegrator, SimpleDynamics
from estimators import *
from sensor import noisy_sensor_mult as sensor

# Define simulation parameters
dt = 0.01  # Time step
T = 1000 # Number of steps
u_max = 1.0

# Obstacle
wall_x = 7.0
goal_x = 17.0

# Initial state (truth)
x_true = jnp.array([5.0, 2.5])  # Start position
goal = jnp.array([goal_x, 5.0])  # Goal position
obstacle = jnp.array([wall_x, 0.0])  # Wall
safe_radius = 0.0  # Safety radius around the obstacle

dynamics = NonlinearSingleIntegrator() 
estimator = GEKF(dynamics, sensor, dt, x_init=x_true)

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
kalman_gains = []
covariances = []

x_traj.append(x_true)
x_estimated, p_estimated = estimator.get_belief()

@jit
def get_b_vector(mu, sigma):

    # Extract the upper triangular elements of a matrix as a 1D array
    upper_triangular_indices = jnp.triu_indices(sigma.shape[0])
    vec_sigma = sigma[upper_triangular_indices]

    b = jnp.concatenate([mu, vec_sigma])

    return b

# Simulation loop
for t in tqdm(range(T), desc="Simulation Progress"):

    belief = get_b_vector(x_estimated, p_estimated)

    # Solve QP
    sol, V, h = solve_qp(belief)

    clf_values.append(V)
    cbf_values.append(h)

    u_opt = sol.primal[0]

    # Apply control to the true state (x_true)
    x_true = x_true + dt * (dynamics.f(x_true) + dynamics.g(x_true) @ u_opt)

    # obtain current measurement
    x_measured =  sensor(x_true, t)

    # updated estimate 
    estimator.predict(u_opt)
    estimator.update(x_measured)
    x_estimated, p_estimated = estimator.get_belief()

    # Store for plotting
    x_traj.append(x_true.copy())
    u_traj.append(u_opt)
    x_meas.append(x_measured)
    x_est.append(x_estimated)
    # print(t, x_true, x_measured)

# Convert to JAX arrays
x_traj = jnp.array(x_traj)

# Conver to numpy arrays for plotting
x_traj = np.array(x_traj)
x_meas = np.array(x_meas)
x_est = np.array(x_est)

# Plot trajectory
plt.figure(figsize=(6, 6))
plt.plot(x_meas[:, 0], x_meas[:, 1], color="Green", linestyle=":", label="Measured Trajectory")
plt.plot(x_traj[:, 0], x_traj[:, 1], "b-", label="Trajectory (True state)")
plt.plot(x_est[:, 0], x_est[:, 1], color="Orange", label="Estimated Trajectory")
plt.scatter(goal[0], goal[1], c="g", marker="*", s=200, label="Goal")

# Plot horizontal line at x = obstacle[0]
plt.axvline(x=obstacle[0], color="r", linestyle="--", label="Obstacle Boundary")

plt.xlabel("x")
plt.ylabel("y")
plt.title("CLF-CBF QP-Controlled Trajectory")
plt.legend()
plt.grid()
plt.show()

# Second figure: X component comparison
plt.figure(figsize=(6, 4))
plt.plot(x_meas[:, 0], color="green", label="Measured x", linestyle="dashed", linewidth=2)
plt.plot(x_est[:, 0], color="orange", label="Estimated x", linestyle="dotted", linewidth=2)
plt.plot(x_traj[:, 0], color="blue", label="True x")
plt.xlabel("Time step")
plt.ylabel("X")
plt.legend()
plt.title("X Trajectory")
plt.show()

# Third figure: Y component comparison
plt.figure(figsize=(6, 4))
plt.plot(x_meas[:, 1], color="green", label="Measured y", linestyle="dashed", linewidth=2)
plt.plot(x_est[:, 1], color="orange", label="Estimated y", linestyle="dotted", linewidth=2)
plt.plot(x_traj[:, 1], color="blue", label="True y")
plt.xlabel("Time step")
plt.ylabel("Y")
plt.legend()
plt.title("Y Trajectory")
plt.show()


# Plot controls
plt.figure(figsize=(6, 4))
plt.plot(cbf_values, color='red', label="CBF")
plt.plot(clf_values, color='green', label="CLF")
plt.plot([u[0] for u in u_traj], color='blue', label="u_x")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("CBF, CLF, and Control Trajectories")
plt.legend()
plt.show()

kalman_gain_traces = [jnp.trace(K) for K in kalman_gains]
covariance_traces = [jnp.trace(P) for P in covariances]

# Plot trace of Kalman gains and covariances
plt.figure(figsize=(6, 4))
plt.plot(kalman_gain_traces, "r-", label="Trace of Kalman Gain")
plt.plot(covariance_traces, "o-", label="Trace of Covariance")
plt.xlabel("Time Step")
plt.ylabel("Trace Value")
plt.title(f"Trace of Kalman Gain and Covariance Over Time ({estimator.name})")
plt.legend()
plt.grid()
plt.show()