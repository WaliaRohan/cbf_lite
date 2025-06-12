import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mplcursors  # for enabling data cursor in matplotlib plots
import numpy as np
from jax import grad, jit
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import clf_1D_doubleint as clf
from dynamics import *
from estimators import *
from sensor import noisy_sensor_mult as sensor

# from sensor import ubiased_noisy_sensor as sensor

# Define simulation parameters
# dt = 0.001 # Time step
# T = 20 # Number of steps
# dt = 1
# T = 5

dt = 0.001
T = 4000 # 5000

u_max = 6.0

# Obstacle
wall_x = 6.0
goal_x = 7.0
x_init = [1.0, 0.0]

# Initial state (truth)
x_true = jnp.array([x_init])  # Start position
goal = jnp.array([goal_x])  # Goal position
obstacle = jnp.array([wall_x])  # Wall

sensor_update_frequency = 0.1 # Hz

dynamics = LinearDoubleIntegrator1D()

# Mean and covariance
mu_u = 0.1
sigma_u = jnp.sqrt(0.01)

mu_v = 0.001
sigma_v = jnp.sqrt(0.0005)

x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v) # mult_noise
# x_initial_measurement = sensor(x_true, t=0, cov=sigma_v) # unbiased_fixed_noise
estimator = GEKF(dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, x_init=x_initial_measurement)
# estimator = EKF(dynamics, dt, x_init=x_initial_measurement, R=sigma_v*jnp.eye(dynamics.state_dim))

# Define belief CBF parameters
n = dynamics.state_dim
alpha = jnp.array([-1.0, 0])
beta = jnp.array([-wall_x])
delta = 0.001  # Probability of failure threshold
cbf = BeliefCBF(alpha, beta, delta, n)

# Control params
clf_gain = 50.0 # CLF linear gain
clf_slack_penalty = 0.0005
# clf_slack_penalty = 5
cbf_gain = 250.0  # CBF linear gain

# Autodiff: Compute Gradients for CLF
grad_V = grad(clf, argnums=0)  # ∇V(x)

# OSQP solver instance
solver = OSQP()

print(jax.default_backend())

list_lgv = []
list_Lg_Lf_h = []
list_rhs = []
list_L_f_h = []
list_L_f_2_h = []
list_grad_h_b = []
list_f_b = []

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
    L_f_hb, L_g_hb, L_f_2_h, Lg_Lf_h, grad_h_b, f_b = cbf.h_dot_b(b, dynamics) # ∇h(x)

    L_f_h = L_f_hb
    L_g_h = L_g_hb

    rhs, L_f_h, h_gain = cbf.h_b_r2_RHS(h, L_f_h, L_f_2_h, cbf_gain)

    # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
    Q = jnp.array([
        [1, 0],
        [0, 2*clf_slack_penalty]
    ])
    
    # c = jnp.zeros(2)  # No linear cost term
    c = jnp.array([-1.0, 0.0])

    A = jnp.array([
        [L_g_V.flatten()[0].astype(float), -1.0], #  LgV u - delta <= -LfV - gamma(V) 
        # [-Lg_Lf_h.flatten()[0].astype(float), 0.0], # -LgLfh u       <= -[alpha1 alpha2].T @ [Lfh h] + Lf^2h
        [1, 0],
        [0, 1]
    ])

    u = jnp.hstack([
        (-L_f_V - clf_gain * V).squeeze(),          # CLF constraint
        # (rhs).squeeze(),                            # CBF constraint: rhs = -[alpha1 alpha2].T [Lfh h] + Lf^2h
        u_max, 
        jnp.inf # no upper limit on slack
    ])

    l = jnp.hstack([
        -jnp.inf, # No lower limit on CLF condition
        # -jnp.inf, # No lower limit on CBF condition
        -u_max,
        0.0 # slack can't be negative
    ])

    # Solve the QP using jaxopt OSQP
    sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    return sol, V, h, L_g_V, Lg_Lf_h, rhs, L_f_h, L_f_2_h, grad_h_b, f_b

x_traj = []  # Store trajectory
x_meas = [] # Measurements
x_est = [] # Estimates
u_traj = []  # Store controls
clf_values = []
cbf_values = []
kalman_gains = []
covariances = []
in_covariances = [] # Innovation Covariance of EKF
prob_leave = [] # Probability of leaving safe set

x_estimated, p_estimated = estimator.get_belief()

x_measured = x_initial_measurement

solve_qp_cpu = jit(solve_qp, backend='cpu')

# Simulation loop
for t in tqdm(range(T), desc="Simulation Progress"):

    x_traj.append(x_true)

    belief = cbf.get_b_vector(x_estimated, p_estimated)

    # Solve QP
    sol, V, h, LgV, Lg_Lf_h, rhs, L_f_h, L_f_2_h, grad_h_b, f_b = solve_qp_cpu(belief)
    # sol, V, h, LgV, Lg_Lf_h, rhs, L_f_h, L_f_2_h, grad_h_b, f_b = solve_qp(belief)

    # DEBUGGING
    list_lgv.append(LgV)
    list_Lg_Lf_h.append(Lg_Lf_h)
    list_rhs.append(rhs)
    list_L_f_h.append(L_f_h)
    list_L_f_2_h.append(L_f_2_h)
    list_grad_h_b.append(grad_h_b)
    list_f_b.append(f_b)

    clf_values.append(V)
    cbf_values.append(h)

    u_sol = jnp.array([sol.primal[0][0]])
    u_opt = jnp.clip(u_sol, -u_max, u_max)


    # Apply control to the true state (x_true)
    x_true = x_true + dt * dynamics.x_dot(x_true, u_opt)

    estimator.predict(u_opt)

    # update measurement and estimator belief
    if t > 0 and t%(1/sensor_update_frequency) == 0:
        # obtain current measurement
        x_measured =  sensor(x_true, t, mu_u, sigma_u, mu_v, sigma_v)
        # x_measured = sensor(x_true) # for identity sensor
        # x_measured = sensor(x_true, t, sigma_v) # for fixed unbiased noise sensor

        if estimator.name == "GEKF":
            estimator.update(x_measured)

        if estimator.name == "EKF":
            estimator.update(x_measured)

        prob_leave.append(estimator.compute_probability_bound(alpha, delta))
    else:
        if len(prob_leave) > 0:
            prob_leave.append(prob_leave[-1])
        else:
            prob_leave.append(-1)

    x_estimated, p_estimated = estimator.get_belief()

    # Store for plotting
    u_traj.append(u_opt)
    x_meas.append(x_measured)
    x_est.append(x_estimated)
    kalman_gains.append(estimator.K)
    covariances.append(p_estimated)
    in_covariances.append(estimator.in_cov)

# Convert to JAX arrays
x_traj = jnp.array(x_traj)

# Convert to numpy arrays for plotting
x_traj = np.array(x_traj).squeeze()
x_meas = np.array(x_meas).squeeze()
x_est = np.array(x_est).squeeze()

time = dt*np.arange(T)  # assuming x_meas.shape[0] == N

# # Plot trajectory with y-values set to zero
# plt.figure(figsize=(6, 6))
# plt.plot(x_meas[:, 0], jnp.zeros_like(x_meas[:, 0]), color="Green", linestyle=":", label="Measured Trajectory")
# plt.plot(x_traj[:, 0], jnp.zeros_like(x_traj[:, 0]), "b-", label="Trajectory (True state)")
# plt.plot(x_est[:, 0], jnp.zeros_like(x_est[:, 0]), "Orange", label="Estimated Trajectory")
# plt.scatter(goal[0], 0, c="g", marker="*", s=200, label="Goal")

# # Plot vertical line at x = obstacle[0]
# plt.axvline(x=obstacle[0], color="r", linestyle="--", label="Obstacle Boundary")

# plt.xlabel("x", fontsize=14)
# plt.ylabel("y (zeroed)", fontsize=14)
# plt.title("1D X-Trajectory (CLF-CBF QP-Controlled)", fontsize=14)
# plt.legend()
# plt.grid()
# plt.show()

# Second figure: X component comparison
plt.figure(figsize=(10, 10))
# Position x
plt.plot(time, x_meas[:, 0], color="green", label="Measured x", linestyle="dashed", linewidth=2, alpha=0.5)
plt.plot(time, x_est[:, 0], color="orange", label="Estimated x", linestyle="dotted", linewidth=1.75)
plt.plot(time, x_traj[:, 0], color="blue", label="True x", linewidth=2)
# Velocity v
plt.plot(time, x_meas[:, 1], color="green", label="Measured v", linestyle="dashdot", linewidth=2, alpha=0.5)
plt.plot(time, x_est[:, 1], color="orange", label="Estimated v", linestyle="dotted", linewidth=1.5)
plt.plot(time, x_traj[:, 1], color="blue", label="True v", linestyle="solid", linewidth=1)
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
plt.ylabel("State value", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.title("State Components Over Time: Position and Velocity", fontsize=18)
plt.grid()
plt.show()

# # Plot controls
plt.figure(figsize=(10, 10))
plt.plot(time, np.array(cbf_values), color='red', label="CBF")
plt.plot(time, np.array(clf_values), color='green', label="CLF")
plt.plot(time, np.array([u[0] for u in u_traj]), color='blue', label="u_x")
plt.xlabel("Time step (s)")
plt.ylabel("Value")
plt.title(f"CBF, CLF, and Control Values ({estimator.name})")
# Tick labels font size
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# Legend font size
plt.legend(fontsize=14)
plt.show()

# Plot CLF (Debug)
# plt.figure(figsize=(6, 4))
# plt.plot(time, np.array(clf_values), color='green', label="CLF")
# plt.plot(time, list_lgv, color='blue', label="LgV")
# plt.xlabel("Time step (s)")
# plt.ylabel("Value")
# plt.title(f"[Debug] CLF and LgV values ({estimator.name})")
# # Tick labels font size
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # Legend font size
# plt.legend(fontsize=14)
# plt.show()

# # Plot CBF (Debug)

# grad_mag = np.array([
#     np.linalg.norm(np.asarray(g).squeeze()) for g in list_grad_h_b
# ])

# f_mag = np.array([
#     np.linalg.norm(np.asarray(f).squeeze()) for f in list_f_b
# ])

# lfh_vals = np.array([
#     (grad@fx).squeeze()
#     for grad, fx in zip(list_grad_h_b, list_f_b)
# ])

# # list_lgv_np = np.array([float(val[0]) for val in list_lgv])
# plt.figure(figsize=(10, 10))
# # plt.plot(time, np.array(cbf_values), color='green', label="CBF")
# # plt.plot(time, list_Lg_Lf_h, color='blue', label="Lg_Lf_h")
# # plt.plot(time, list_rhs, color='red', label='rhs')
# plt.plot(time, list_L_f_h, color='purple', label="L_f_h")
# mplcursors.cursor() 
# # plt.plot(time, list_L_f_2_h, color='black', label="L_f_2_h")
# plt.plot(time, grad_mag, color='orange', label="|grad_h_b|")
# plt.plot(time, f_mag, color='maroon', label="|f_b|")
# plt.plot(time, lfh_vals, color = 'yellow', label="product")
# plt.xlabel("Time step (s)")
# plt.ylabel("Value")
# plt.title(f"[Debug] CBF and Lg_Lf_h values ({estimator.name})")
# # Tick labels font size
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# # Legend font size
# plt.legend(fontsize=14)
# plt.show()

kalman_gain_traces = [jnp.trace(K) for K in kalman_gains]
covariance_traces = [jnp.trace(P) for P in covariances]
inn_cov_traces = [jnp.trace(cov) for cov in in_covariances]

# # Plot trace of Kalman gains and covariances
plt.figure(figsize=(10, 10))
plt.plot(time, np.array(kalman_gain_traces), "b-", label="Trace of Kalman Gain")
plt.plot(time, np.array(covariance_traces), "r-", label="Trace of Covariance")
# plt.plot(time, np.array(inn_cov_traces), "g-", label="Trace of Innovation Covariance")
# plt.plot(time, np.array(prob_leave), "purple", label="P_leave")
plt.xlabel("Time Step (s)")
plt.ylabel("Trace Value")
plt.title(f"Trace of Kalman Gain and Covariance Over Time ({estimator.name})")
plt.legend()
plt.grid()
plt.show()

## Probability of leaving safe set


# # Plot distance from obstacle

# dist = wall_x - x_est

# plt.figure(figsize=(10, 10))
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
print(f"Sensor Update Frequency (Hz): {sensor_update_frequency}")

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