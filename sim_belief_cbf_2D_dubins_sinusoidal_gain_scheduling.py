import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# import mplcursors  # for enabling data cursor in matplotlib plots
import numpy as np
from jax import grad, jit
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import gain_schedule_ctrl, sinusoidal_trajectory, update_trajectory_index, s_trajectory, straight_trajectory
from dynamics import *
from estimators import *
from sensor import noisy_sensor_mult as sensor
# from sensor import ubiased_noisy_sensor as sensor

# Sim Params
dt = 0.001
T = 10000# 5000
dynamics = DubinsMultCtrlDynamics()

# Sensor Params
mu_u = 0.1
sigma_u = jnp.sqrt(0.01) # Standard deviation
mu_v = 0.001
sigma_v = jnp.sqrt(0.0005) # Standard deviation
sensor_update_frequency = 0.1 # Hz

# Obstacle
wall_y = 10.0

# Initial state
lin_vel = 1.0
x_init = [0.0, 0.0, 0.1] # x, y, v, theta

# Initial state (truth)
x_true = jnp.array(x_init)  # Start position
goal = 18.0*jnp.array([1.0, 1.0])  # Goal position
obstacle = jnp.array([wall_y])  # Wall

# Mean and covariance
x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v) # mult_noise
# x_initial_measurement = sensor(x_true, t=0, std=sigma_v) # unbiased_fixed_noise
# Observation function: Return second and 4rth element of the state vector
# self.h = lambda x: x[jnp.array([1, 3])]
h = lambda x: jnp.array([x[1]])
estimator = GEKF(dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, h=h, x_init=x_initial_measurement)
# estimator = EKF(dynamics, dt, h=h, x_init=x_initial_measurement, R=jnp.square(sigma_v)*jnp.eye(dynamics.state_dim))

# Define belief CBF parameters
n = dynamics.state_dim
alpha = jnp.array([0.0, -1.0, 0.0])
beta = jnp.array([-wall_y])
delta = 0.001  # Probability of failure threshold
cbf = BeliefCBF(alpha, beta, delta, n)

# CBF 2
alpha2 = jnp.array([0.0, 1.0, 0.0, 0.0])
beta2 = jnp.array([-wall_y])
delta2 = 0.001  # Probability of failure threshold
cbf2 = BeliefCBF(alpha2, beta2, delta2, n)

# Control params
MAX_LINEAR=lin_vel
MAX_ANGULAR = 0.25
U_MAX = np.array([MAX_LINEAR, MAX_ANGULAR])
clf_gain = 20.0 # CLF linear gain
clf_slack_penalty = 50.0
cbf_gain = 10.0  # CBF linear gain
CBF_ON = True

# OSQP solver instance
solver = OSQP()

print(jax.default_backend())

m = len(U_MAX) # control dim
var_dim = m + 1 # ctrl dim + slack variable

@jit
def solve_qp(b, goal_loc):

    x_estimated, sigma = cbf.extract_mu_sigma(b)

    u_nom = gain_schedule_ctrl(v_r=lin_vel,
                               x = x_estimated,
                               x_d = goal_loc,
                               lambda1=1.0, a1=10.0, a2=15.0)

    # # Compute CBF components
    # h = cbf.h_b(b)
    # L_f_hb, L_g_hb, L_f_2_h, Lg_Lf_h, grad_h_b, f_b = cbf.h_dot_b(b, dynamics) # ∇h(x)

    # L_f_h = L_f_hb

    # rhs, L_f_h, h_gain = cbf.h_b_r2_RHS(h, L_f_h, L_f_2_h, cbf_gain)

    # # Compute CBF2 components
    # h_2 = cbf2.h_b(b)
    # L_f_hb_2, L_g_hb_2, L_f_2_h_2, Lg_Lf_h_2, _, _ = cbf2.h_dot_b(b, dynamics) # ∇h(x)

    # L_f_h_2 = L_f_hb_2

    # rhs2, L_f_h2, _ = cbf2.h_b_r2_RHS(h_2, L_f_h_2, L_f_2_h_2, cbf_gain)

    # A = jnp.vstack([
    #     jnp.concatenate([-Lg_Lf_h, jnp.array([0.0, 0.0])]), # -LgLfh u       <= -[alpha1 alpha2].T @ [Lfh h] + Lf^2h
    #     jnp.concatenate([-Lg_Lf_h_2, jnp.array([0.0, 0.0])]), # 2nd CBF
    #     jnp.eye(var_dim)
    # ])

    # u = jnp.hstack([
    #     (rhs).squeeze(),                            # CBF constraint: rhs = -[alpha1 alpha2].T [Lfh h] + Lf^2h
    #     (rhs2).squeeze(),                           # 2nd CBF constraint
    #     U_MAX, 
    #     jnp.inf # no upper limit on slack
    # ])

    # l = jnp.hstack([
    #     -jnp.inf, # No lower limit on CBF condition
    #     -jnp.inf, # 2nd CBF
    #     -U_MAX,
    #     0.0 # slack can't be negative
    # ])

    # if not CBF_ON:
    #     A = jnp.concatenate([A[:0], A[2:]], axis=0) # Remove CBF conditions
    #     u = jnp.concatenate([u[:0], u[2:]], axis=0)        # Remove corresponding element in u
    #     l = jnp.concatenate([l[:0], l[2:]], axis=0)         # Remove corresponding element in l

    # # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
    # Q = jnp.eye(var_dim)
    # Q = Q.at[-1, -1].set(2*clf_slack_penalty)

    # c = jnp.append(-2.0*u_nom.flatten(), 0.0)

    # # Solve the QP using jaxopt OSQP
    # sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    # return sol, V

    return u_nom

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

x_nom = [] # Store nominal trajectory

x_estimated, p_estimated = estimator.get_belief()

x_measured = x_initial_measurement

solve_qp_cpu = jit(solve_qp, backend='cpu')

goal_loc = x_init

t_vec = jnp.arange(0.0, T + 1.0, 1.0)*dt
# goal_x_nom = sinusoidal_trajectory(t_vec, A=goal[1], omega=1.0, v=lin_vel).T  # shape (T/dt, 2)
goal_x_nom = straight_trajectory(t_vec, y_val=0.5, lin_v=lin_vel).T
# goal_x_nom = s_trajectory(t_vec, A=5.0, omega=0.5, v=lin_vel).T

# plt.figure(figsize=(10, 10))
# plt.plot(goal_x_nom[:, 0], goal_x_nom[:, 1], "Green", label="Nominal Trajectory")
# plt.show()
# plt.pause(0)

traj_idx = 0
goal_loc = goal_x_nom[traj_idx]

# Simulation loop
for t in tqdm(range(T), desc="Simulation Progress"):

    x_traj.append(x_true)

    belief = cbf.get_b_vector(x_estimated, p_estimated)

    # target_goal_loc = sinusoidal_trajectory(t*dt, A=goal[1], omega=1.0, v=lin_vel)

    sol = solve_qp_cpu(belief, goal_loc)
    # sol, V = solve_qp(belief, goal_loc)

    # clf_values.append(V)
    # cbf_values.append(h)

    # u_sol = jnp.array([sol.primal[0][0]])

    u_sol = sol
    u_opt = jnp.clip(u_sol, -U_MAX, U_MAX)
    # u_opt = u_sol

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

        # prob_leave.append(estimator.compute_probability_bound(alpha, delta))
    # else:
    #     if len(prob_leave) > 0:
    #         prob_leave.append(prob_leave[-1])
    #     else:
    #         prob_leave.append(-1)

    x_estimated, p_estimated = estimator.get_belief()


    # if (x_estimated[1] < wall_y and x_estimated[1] > -wall_y):
    #     eta = 7.0
    # else:
    #     eta = 15.0

    eta=1.0

    # traj_idx = update_trajectory_index(x_estimated[0:2], goal_x_nom[:, 0:2], traj_idx, eta=eta)
    goal_loc = goal_x_nom[t]

    # Store for plotting
    u_traj.append(u_opt)
    x_meas.append(x_measured)
    x_est.append(x_estimated)
    kalman_gains.append(estimator.K)
    covariances.append(p_estimated)
    in_covariances.append(estimator.in_cov)
    x_nom.append(goal_loc[:2])

# Convert to JAX arrays
x_traj = jnp.array(x_traj)

# Convert to numpy arrays for plotting
x_traj = np.array(x_traj).squeeze()
x_meas = np.array(x_meas).squeeze()
x_est = np.array(x_est).squeeze()
u_traj = np.array(u_traj)

x_nom = np.array(x_nom).squeeze()
np.savetxt("x_nom.csv", x_nom, delimiter=",")

time = dt*np.arange(T)  # assuming x_meas.shape[0] == N

# Plot trajectory with y-values set to zero
plt.figure(figsize=(10, 10))
plt.plot(x_meas[:, 0], x_meas[:, 1], color="Green", linestyle=":", label="Measured Trajectory", alpha=0.5)
plt.plot(x_traj[:, 0], x_traj[:, 1], "b-", label="Trajectory (True state)")
plt.plot(x_est[:, 0], x_est[:, 1], "Orange", label="Estimated Trajectory")
plt.plot(x_nom[:, 0], x_nom[:, 1], "Green", label="Nominal Trajectory")
# plt.axhline(y=wall_y, color="red", linestyle="dashed", linewidth=1, label="Obstacle")
# plt.axhline(y=-wall_y, color="red", linestyle="dashed", linewidth=1)
# plt.axhline(y=goal[1], color="purple", linestyle="dashed", linewidth=1, label="Goal")
# plt.scatter(goal[0], goal[1], c="g", marker="*", s=200, label="Goal")
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)
plt.title("2D X-Trajectory (CLF-CBF QP-Controlled)", fontsize=14)
plt.legend()
plt.grid()
# plt.show()

# Plot controls
plt.figure(figsize=(10, 10))
# plt.plot(time, np.array(cbf_values), color='red', label="CBF")
# plt.plot(time, np.array(clf_values), color='green', label="CLF")
for i in range(m):
    plt.plot(time, u_traj[:, i], label=f"u_{i}")
plt.xlabel("Time step (s)")
plt.ylabel("Control value")
plt.title(f"Control Values ({estimator.name})")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
# plt.show()


# # Plot CLF (Debug)
# plt.figure(figsize=(10, 10))
# plt.plot(time, np.array(clf_values), color='green', label="CLF")
# # plt.plot(time, list_lgv, color='blue', label="LgV")
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

# dist = wall_y - x_est

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
# print(f"Control Input Max (u_max): {u_max}")
print(f"Sensor Update Frequency (Hz): {sensor_update_frequency}")

print("\n--- Environment Setup ---")
print(f"Obstacle Position (wall_y): {wall_y}")
print(f"Goal Position (goal_x): {goal}")
print(f"Initial Position (x_init): {x_init}")

print("\n--- Belief CBF Parameters ---")
print(f"Failure Probability Threshold (delta): {delta}")

print("\n--- Control Parameters ---")
print(f"CLF Linear Gain (clf_gain): {clf_gain}")
print(f"CLF Slack (clf_slack): {clf_slack_penalty}")
print(f"CBF Linear Gain (cbf_gain): {cbf_gain}")

# Print Metrics

print("\n--- Results ---")

print("Number of estimate exceedances: ", np.sum(x_est > wall_y))
print("Number of true exceedences", np.sum(x_traj > wall_y))
print("Max estimate value: ", np.max(x_est))
print("Max true value: ", np.max(x_traj))
print("Mean true distance from obstacle: ", np.mean(wall_y - x_est))
print("Average controller effort: ", np.linalg.norm(u_traj, ord=2))
print("Cummulative distance to goal: ", np.sum(np.abs(x_traj - wall_y)))
print(f"{estimator.name} Tracking RMSE: ", np.sqrt(np.mean((x_traj - x_est) ** 2)))


# Plot distance from safety boudary of estimates, max estimate value, 