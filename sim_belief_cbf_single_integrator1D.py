import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import vanilla_clf_x as clf
from dynamics import *
from estimators import *
from sensor import noisy_sensor_mult as sensor

# Define simulation parameters
dt = 0.001 # Time step
T = 5000 # Number of steps
u_max = 10.0

# Obstacle
wall_x = 5.0
goal_x = 6.0
x_init = 1.0

# Initial state (truth)
x_true = jnp.array([x_init])  # Start position
goal = jnp.array([goal_x])  # Goal position
obstacle = jnp.array([wall_x])  # Wall

dynamics = NonLinearSingleIntegrator1D() 

# High noise
mu_u = 0.1
sigma_u = jnp.sqrt(0.001)

mu_v = 0.01
sigma_v = jnp.sqrt(0.0005)

# Low noise
# mu_u = 0.0174
# sigma_u = jnp.sqrt(2.916e-4) # 10 times more than what was shown in GEKF paper

# # Additive noise
# mu_v = -0.0386
# sigma_v = jnp.sqrt(7.97e-5)

x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v)
estimator = GEKF(dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, x_init=x_initial_measurement)
# estimator = EKF(dynamics, dt, x_init=x_initial_measurement, R=sigma_v*jnp.eye(dynamics.state_dim))

# Define belief CBF parameters
n = dynamics.state_dim
alpha = jnp.array([-1.0])
beta = jnp.array([-wall_x])
delta = 0.001  # Probability of failure threshold
cbf = BeliefCBF(alpha, beta, delta, n)

# Control params
clf_gain = 20.0  # CLF linear gain
clf_slack_penalty = 0.00025
# clf_slack_penalty = 100.0
cbf_gain = 200.0  # CBF linear gain

# Autodiff: Compute Gradients for CLF and CBF
grad_V = grad(clf, argnums=0)  # ∇V(x)

# OSQP solver instance
solver = OSQP()

print(jax.default_backend())

# @jit
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

    # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
    Q = jnp.array([
        [1, 0],
        [0, 2*clf_slack_penalty]
    ])
    
    c = jnp.zeros(2)  # No linear cost term

    A = jnp.array([
        [L_g_V.flatten()[0].astype(float), -1.0], # -Lgh u         <=  Lfh + alpha(h)
        [-L_g_h.flatten()[0].astype(float), 0.0], #  LgV u - delta <= -LfV - gamma(V)
        [1, 0],
        [0, 1]
    ])

    u = jnp.hstack([
        (-L_f_V - clf_gain * V).squeeze(),          # CLF constraint
        (L_f_h.squeeze() + cbf_gain * h).squeeze(), # CBF constraint
        u_max, 
        jnp.inf # no upper limit on slack
    ])

    l = jnp.hstack([
        -jnp.inf, # No lower limit on CLF condition
        -jnp.inf, # No lower limit on CBF condition
        -u_max,
        0.0 # slack can't be negative
    ])

    # Solve the QP using jaxopt OSQP
    sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    return sol, clf_gain*V, cbf_gain*h

x_traj = []  # Store trajectory
x_meas = [] # Measurements
x_est = [] # Estimates
u_traj = []  # Store controls
clf_values = []
cbf_values = []
kalman_gains = []
covariances = []

x_estimated, p_estimated = estimator.get_belief()

# @jit
def get_b_vector(mu, sigma):

    # Extract the upper triangular elements of a matrix as a 1D array
    upper_triangular_indices = jnp.triu_indices(sigma.shape[0])
    vec_sigma = sigma[upper_triangular_indices]

    b = jnp.concatenate([mu, vec_sigma])

    return b

x_measured = x_initial_measurement

solve_qp_cpu = jit(solve_qp, backend='cpu')


# Simulation loop
for t in tqdm(range(T), desc="Simulation Progress"):

    x_traj.append(x_true)

    belief = get_b_vector(x_estimated, p_estimated)

    # Solve QP
    sol, V, h = solve_qp_cpu(belief)

    clf_values.append(V)
    cbf_values.append(h)

    u_opt = jnp.array([sol.primal[0][0]])

    # Apply control to the true state (x_true)
    x_true = x_true + dt * (dynamics.f(x_true) + dynamics.g(x_true) @ u_opt)

    estimator.predict(u_opt)

    # update measurement and estimator belief
    if t > 0 and t%10 == 0:
        # obtain current measurement
        x_measured =  sensor(x_true, t, mu_u, sigma_u, mu_v, sigma_v)

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

time = dt*np.arange(T)  # assuming x_meas.shape[0] == N

# Plot trajectory with y-values set to zero
plt.figure(figsize=(6, 6))
plt.plot(x_meas[:, 0], jnp.zeros_like(x_meas[:, 0]), color="Green", linestyle=":", label="Measured Trajectory")
plt.plot(x_traj[:, 0], jnp.zeros_like(x_traj[:, 0]), "b-", label="Trajectory (True state)")
plt.plot(x_est[:, 0], jnp.zeros_like(x_est[:, 0]), "Orange", label="Estimated Trajectory")
plt.scatter(goal[0], 0, c="g", marker="*", s=200, label="Goal")

# Plot vertical line at x = obstacle[0]
plt.axvline(x=obstacle[0], color="r", linestyle="--", label="Obstacle Boundary")

plt.xlabel("x", fontsize=14)
plt.ylabel("y (zeroed)", fontsize=14)
plt.title("1D X-Trajectory (CLF-CBF QP-Controlled)", fontsize=14)
plt.legend()
plt.grid()
plt.show()

# Second figure: X component comparison
plt.figure(figsize=(10, 10))
plt.plot(time, x_meas[:, 0], color="green", label="Measured x", linestyle="dashed", linewidth=2, alpha=0.5)
plt.plot(time, x_est[:, 0], color="orange", label="Estimated x", linestyle="dotted", linewidth=6)
plt.plot(time, x_traj[:, 0], color="blue", label="True x")
# Add horizontal lines
plt.axhline(y=wall_x, color="red", linestyle="dashed", linewidth=1, label="Obstacle")
plt.axhline(y=goal_x, color="purple", linestyle="dashed", linewidth=1, label="Goal")
# # Compute 2-sigma bounds (99% confidence interval)
# cov = np.abs(covariances).squeeze(2)
# cov_std = 2 * np.sqrt(cov) # Since covariances is 1D
# # Plot 2-sigma confidence interval
# plt.fill_between(range(len(x_est)), (x_est - cov_std).squeeze(), (x_est + cov_std).squeeze(), 
#                  color="cyan", alpha=0.3, label="95% confidence interval")
plt.xlabel("Time step (s)", fontsize=16)
plt.ylabel("x (m)", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.title("X position (m)", fontsize=18)
plt.show()

# # Plot controls
# plt.figure(figsize=(6, 4))
# plt.plot(time, np.array(cbf_values), color='red', label="CBF")
# plt.plot(time, np.array(clf_values), color='green', label="CLF")
# plt.plot(time, np.array([u[0] for u in u_traj]), color='blue', label="u_x")
# plt.xlabel("Time step (s)")
# plt.ylabel("Value")
# plt.title(f"CBF, CLF, and Control Values ({estimator.name})")
# # Tick labels font size
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # Legend font size
# plt.legend(fontsize=14)
# plt.show()

# kalman_gain_traces = [jnp.trace(K) for K in kalman_gains]
# covariance_traces = [jnp.trace(P) for P in covariances]

# # Plot trace of Kalman gains and covariances
# plt.figure(figsize=(6, 4))
# plt.plot(time, np.array(kalman_gain_traces), "b-", label="Trace of Kalman Gain")
# plt.plot(time, np.array(covariance_traces), "r-", label="Trace of Covariance")
# plt.xlabel("Time Step (s)")
# plt.ylabel("Trace Value")
# plt.title(f"Trace of Kalman Gain and Covariance Over Time ({estimator.name})")
# plt.legend()
# plt.grid()
# plt.show()


# # Plot distance from obstacle

# dist = wall_x - x_est

# plt.figure(figsize=(6, 4))
# plt.plot(time, dist[:, 0], color="red", linestyle="dashed")
# plt.title(f"Distance from safe boundary ({estimator.name})")
# plt.xlabel("Time Step (s)")
# plt.ylabel("Distance")
# plt.legend()
# plt.grid()
# plt.show()


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
print(f"CLF Slack (clf_slack): {clf_slack_penalty}")
print(f"CBF Linear Gain (cbf_gain): {cbf_gain}")

# Print Metrics

print("\n--- Results ---")

print("Number of estimate exceedances: ", np.sum(x_est > wall_x))
print("Number of true exceedences", np.sum(x_traj > wall_x))
print("Max estimate value: ", np.max(x_est))
print("Max true value: ", np.max(x_traj))
print("Mean true distance from obstacle: ", np.mean(wall_x - x_est))
print("Average controller effort: ", np.linalg.norm(u_traj, ord=2))
print("Cummulative distance to goal: ", np.sum(np.abs(x_traj - wall_x)))
print(f"{estimator.name} Tracking RMSE: ", np.sqrt(np.mean((x_traj - x_est) ** 2)))


# Plot distance from safety boudary of estimates, max estimate value, 