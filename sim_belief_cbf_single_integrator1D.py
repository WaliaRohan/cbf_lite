import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import vanilla_clf_x as clf
from dynamics import SingleIntegrator1D
from estimators import *
from sensor import noisy_sensor_mult as sensor

# Define simulation parameters
dt = 0.001 # Time step
T = 5000 # Number of steps
u_max = 1.0

# Obstacle
wall_x = 7.0
goal_x = 10.0
x_init = 5.0

# Initial state (truth)
x_true = jnp.array([x_init])  # Start position
goal = jnp.array([goal_x])  # Goal position
obstacle = jnp.array([wall_x])  # Wall
safe_radius = 0.0  # Safety radius around the obstacle

dynamics = SingleIntegrator1D() 

x_initial_measurement = sensor(x_true, 0)
estimator = GEKF(dynamics, sensor, dt, x_init=x_initial_measurement, R=jnp.sqrt(0.0001)*jnp.eye(dynamics.state_dim))

# Define belief CBF parameters
n = dynamics.state_dim
alpha = jnp.array([-1.0])  # Example matrix
beta = jnp.array([-wall_x])  # Example vector
delta = 0.001  # Probability of failure threshold
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
    h = cbf.h_b(b)
    L_f_hb, L_g_hb = cbf.h_dot_b(b, dynamics) # ∇h(x)

    L_f_h = L_f_hb
    L_g_h = L_g_hb

    # Define QP matrices
    Q = jnp.eye(dynamics.state_dim)  # Minimize ||u||^2
    c = jnp.zeros(dynamics.state_dim)  # No linear cost term

    A = jnp.vstack([
        L_g_V,   # CLF constraint
        -L_g_h,   # CBF constraint (negated for inequality direction)
        jnp.eye(dynamics.state_dim)
    ])

    u = jnp.hstack([
        (-L_f_V - clf_gain * V + clf_slack).squeeze(),   # CLF constraint
        (L_f_h.squeeze() + cbf_gain * h).squeeze(),     # CBF constraint
        u_max 
    ])

    l = jnp.hstack([
        -jnp.inf,
        -jnp.inf,
        -u_max
    ])

    # Solve the QP using jaxopt OSQP
    sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    return sol, V, h

x_traj = []  # Store trajectory
x_meas = [] # Measurements
x_est = [] # Estimates
u_traj = []  # Store controls
clf_values = []
cbf_values = []
kalman_gains = []
covariances = []

# num_steps = int(T/dt)
# state_dim = dynamics.state_dim
# control_dim = dynamics.g_matrix.shape[1]

# x_traj = np.zeros((num_steps, state_dim))  # Assuming 'state_dim' is known
# x_meas = np.zeros((num_steps, state_dim))
# x_est = np.zeros((num_steps, state_dim))
# u_traj = np.zeros((num_steps, control_dim))  # Assuming 'control_dim' is known
# clf_values = np.zeros(num_steps)
# cbf_values = np.zeros(num_steps)
# kalman_gains = np.zeros((num_steps, 1))  # Adjust shape as needed
# covariances = np.zeros((num_steps, 1))  # Assuming covariance is square

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

    # x_traj[t] = x_true.copy()
    x_traj.append(x_true)

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
    # update measurement every 10 iteration steps
    if t > 0 and t%2 == 0:
        if estimator.name == "GEKF":
            estimator.update_1D(x_measured)

        if estimator.name == "EKF":
            estimator.update(x_measured)

    x_estimated, p_estimated = estimator.get_belief()

    # Store for plotting
    u_traj.append(u_opt)
    x_meas.append(x_measured)
    x_est.append(x_estimated)
    kalman_gains.append(estimator.K)
    covariances.append(p_estimated)


# Convert to JAX arrays
x_traj = jnp.array(x_traj)

# Conver to numpy arrays for plotting
x_traj = np.array(x_traj)
x_meas = np.array(x_meas)
x_est = np.array(x_est)

# Plot trajectory with y-values set to zero
plt.figure(figsize=(6, 6))
plt.plot(x_meas[:, 0], jnp.zeros_like(x_meas[:, 0]), color="Green", linestyle=":", label="Measured Trajectory")
plt.plot(x_traj[:, 0], jnp.zeros_like(x_traj[:, 0]), "b-", label="Trajectory (True state)")
plt.plot(x_est[:, 0], jnp.zeros_like(x_est[:, 0]), "Orange", label="Estimated Trajectory")
plt.scatter(goal[0], 0, c="g", marker="*", s=200, label="Goal")

# Plot vertical line at x = obstacle[0]
plt.axvline(x=obstacle[0], color="r", linestyle="--", label="Obstacle Boundary")

plt.xlabel("x")
plt.ylabel("y (zeroed)")
plt.title("1D X-Trajectory (CLF-CBF QP-Controlled)")
plt.legend()
plt.grid()
plt.show()

# Second figure: X component comparison
plt.figure(figsize=(6, 4))
plt.plot(x_meas[:, 0], color="green", label="Measured x", linestyle="dashed", linewidth=2)
plt.plot(x_est[:, 0], color="orange", label="Estimated x", linestyle="dotted", linewidth=6)
plt.plot(x_traj[:, 0], color="blue", label="True x")
# Add horizontal lines
plt.axhline(y=wall_x, color="red", linestyle="dashed", linewidth=1, label="Obstacle")
plt.axhline(y=goal_x, color="purple", linestyle="dashed", linewidth=1, label="Goal")
# Compute 2-sigma bounds (99% confidence interval)
cov = np.abs(covariances).squeeze(2)
cov_std = 2 * np.sqrt(cov) # Since covariances is 1D
# Plot 2-sigma confidence interval
plt.fill_between(range(len(x_est)), (x_est - cov_std).squeeze(), (x_est + cov_std).squeeze(), 
                 color="cyan", alpha=0.3, label="95% confidence interval")
plt.xlabel("Time step")
plt.xlabel("Time step")
plt.ylabel("X")
plt.legend()
plt.title("X Trajectory")
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
plt.plot(kalman_gain_traces, "b-", label="Trace of Kalman Gain")
plt.plot(covariance_traces, "r-", label="Trace of Covariance")
plt.xlabel("Time Step")
plt.ylabel("Trace Value")
plt.title(f"Trace of Kalman Gain and Covariance Over Time ({estimator.name})")
plt.legend()
plt.grid()
plt.show()


# Plot distance from obstacle

dist = wall_x - x_est

plt.figure(figsize=(6, 4))
plt.plot(dist[:, 0], color="red", linestyle="dashed")
plt.title(f"{estimator.name} Distance from safe boundary")
plt.xlabel("Time Step")
plt.ylabel("Distance")
plt.legend()
plt.grid()
plt.show()


# Print Sim Params

print("\n--- Simulation Parameters ---")

print(dynamics.name)
print(estimator.name)
print(f"Time Step (dt): {dt}")
print(f"Number of Steps (T): {T}")
print(f"Control Input Max (u_max): {u_max}")

print("\n--- Environment Setup ---")
print(f"Obstacle Position (wall_x): {wall_x}")
print(f"Goal Position (goal_x): {goal_x}")
print(f"Initial Position (x_init): {x_init}")

print("\n--- Belief CBF Parameters ---")
print(f"Failure Probability Threshold (delta): {delta}")

print("\n--- Control Parameters ---")
print(f"CLF Linear Gain (clf_gain): {clf_gain}")
print(f"CLF Slack (clf_slack): {clf_slack}")
print(f"CBF Linear Gain (cbf_gain): {cbf_gain}")

# Print Metrics

print("\n--- Results ---")

print("Number of exceedances: ", np.sum(x_traj > wall_x))
print("Max True value: ", np.max(x_traj))
print("Max estimate value: ", np.max(x_est))
print("Mean difference from obstacle: ", np.mean(wall_x - x_est))
print(f"{estimator.name} Tracking RMSE: ", np.sqrt(np.mean((x_traj - x_est) ** 2)))

# Plot distance from safety boudary of estimates, max estimate value, 