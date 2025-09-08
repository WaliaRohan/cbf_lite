import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# import mplcursors  # for enabling data cursor in matplotlib plots
import numpy as np
from jax import jit
from jaxopt import BoxOSQP as OSQP
from tqdm import tqdm

from cbfs import BeliefCBF
from cbfs import gain_schedule_ctrl, s_trajectory
from dynamics import *
from estimators import *
from sensor import noisy_sensor_mult as sensor
from scipy.stats import chi2
import json

# from sensor import ubiased_noisy_sensor as sensor

# Sim Params
dt = 0.001
T = 30000 # EKF BECOMES UNSTABLE > 30000 !!! 
Q_scale_factor = 0.001
dynamics =  DubinsMultCtrlDynamics(
Q = jnp.diag(jnp.array([(0.1*Q_scale_factor)**2,
                        (0.1*Q_scale_factor)**2,
                        (0.005*Q_scale_factor)**2,
                        (0.005*Q_scale_factor)**2]))) # UnicyleDynamics()

# Sensor Params
mu_u = 0.1
sigma_u = jnp.sqrt(0.01) # Standard deviation
mu_v = 0.001
sigma_v = jnp.sqrt(0.0005) # Standard deviation
sensor_update_frequency = 0.1 # Hz

# Obstacle
wall_y = 5.0

# Initial state
# x_init = [0.0, 0.0, 0.8] # x, y, v, theta
lin_vel = 5.0
x_init = [0.0, 0.0, lin_vel, 0.8]

# Initial state (truth)
x_true = jnp.array(x_init)  # Start position
goal = 18.0*jnp.array([1.0, 1.0])  # Goal position
obstacle = jnp.array([wall_y])  # Wall

# Mean and covariance
x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v) # mult_noise
# x_initial_measurement = sensor(x_true, t=0, std=sigma_v) # unbiased_fixed_noise
# Observation function: Return second and 4rth element of the state vector
# self.h = lambda x: x[jnp.array([1, 3])]
obs_fun = lambda x: jnp.array([x[1]])
estimator = GEKF(dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, h=obs_fun, x_init=x_initial_measurement)
# estimator = EKF(dynamics, dt, h=obs_fun, x_init=x_initial_measurement, R=jnp.square(sigma_v)*jnp.eye(dynamics.state_dim))

# Define belief CBF parameters
n = dynamics.state_dim
# alpha = jnp.array([0.0, -1.0, 0.0])
alpha = jnp.array([0.0, -1.0, 0.0, 0.0])
beta = jnp.array([-wall_y])
delta = 0.001  # Probability of failure threshold
cbf = BeliefCBF(alpha, beta, delta, n)

# CBF 2
# alpha2 = jnp.array([0.0, 1.0, 0.0])
alpha2 = jnp.array([0.0, 1.0, 0.0, 0.0])
beta2 = jnp.array([-wall_y])
delta2 = 0.001  # Probability of failure threshold
cbf2 = BeliefCBF(alpha2, beta2, delta2, n)

# Control params
MAX_LINEAR=lin_vel
MAX_ANGULAR = 0.5
U_MAX = np.array([MAX_LINEAR, MAX_ANGULAR])
clf_gain = 20.0 # CLF linear gain
clf_slack_penalty = 50.0
cbf_gain = 50.0  # CBF linear gain
CBF_ON = True

# OSQP solver instance
solver = OSQP()

m = len(U_MAX) # control dim
var_dim = m + 1 # ctrl dim + slack variable

# @jit
def solve_qp(b, goal_loc):

    x_estimated, sigma = cbf.extract_mu_sigma(b)

    x_current = jnp.concatenate([x_estimated[:2], x_estimated[3:]]) # Deleting velocity fro mstate vector to match with goal_loc
    u_nom = gain_schedule_ctrl(v_r=lin_vel,
                               x = x_current ,
                               ell=0.163,
                               x_d = goal_loc,
                               lambda1=1.0, a1=16.0, a2=100.0)

    # Compute CBF components
    h = cbf.h_b(b)
    L_f_hb, L_g_hb, L_f_2_h, Lg_Lf_h, grad_h_b, f_b = cbf.h_dot_b(b, dynamics) # ∇h(x)

    L_f_h = L_f_hb

    rhs, L_f_h, h_gain = cbf.h_b_r2_RHS(h, L_f_h, L_f_2_h, cbf_gain)

    # Compute CBF2 components
    h_2 = cbf2.h_b(b)
    L_f_hb_2, L_g_hb_2, L_f_2_h_2, Lg_Lf_h_2, _, _ = cbf2.h_dot_b(b, dynamics) # ∇h(x)

    L_f_h_2 = L_f_hb_2

    rhs2, L_f_h2, _ = cbf2.h_b_r2_RHS(h_2, L_f_h_2, L_f_2_h_2, cbf_gain)

    A = jnp.vstack([
        jnp.concatenate([-Lg_Lf_h, jnp.array([0.0])]), # -LgLfh u       <= -[alpha1 alpha2].T @ [Lfh h] + Lf^2h
        jnp.concatenate([-Lg_Lf_h_2, jnp.array([0.0])]), # 2nd CBF
        jnp.eye(var_dim)
    ])

    u = jnp.hstack([
        (rhs).squeeze(),                            # CBF constraint: rhs = -[alpha1 alpha2].T [Lfh h] + Lf^2h
        (rhs2).squeeze(),                           # 2nd CBF constraint
        U_MAX, 
        jnp.inf # no upper limit on slack
    ])

    l = jnp.hstack([
        -jnp.inf, # No lower limit on CBF condition
        -jnp.inf, # 2nd CBF
        -U_MAX,
        0.0 # slack can't be negative
    ])

    if CBF_ON:
        A, u, l = A, u, l
    else:
        A, u, l = A[2:], u[2:], l[2:]


    # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
    Q = jnp.eye(var_dim)
    Q = Q.at[-1, -1].set(2*clf_slack_penalty)

    c = jnp.append(-2.0*u_nom.flatten(), 0.0)

    # Solve the QP using jaxopt OSQP
    sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    return sol, h, h_2

    # return u_nom

x_traj = []  # Store trajectory
x_meas = [] # Measurements
x_est = [] # Estimates
u_traj = []  # Store controls
clf_values = []
cbf_values = []
kalman_gains = []
covariances = []
in_covariances = [] # Innovation Covariance of EKF
meas_matrices = [] # The "C" in Kalman gain
prob_leave = [] # Probability of leaving safe set
NEES_list = []
NIS_list = []


x_nom = [] # Store nominal trajectory

x_estimated, p_estimated = estimator.get_belief()

x_measured = x_initial_measurement

solve_qp_cpu = jit(solve_qp, backend='cpu')

goal_loc = x_init

t_vec = jnp.arange(0.0, T + 1.0, 1.0)*dt
# goal_x_nom = sinusoidal_trajectory(t_vec, A=goal[1], omega=1.0, v=lin_vel).T  # shape (T/dt, 2)
# goal_x_nom = straight_trajectory(t_vec, y_val=0.5, lin_v=lin_vel).T
goal_x_nom = s_trajectory(t_vec, A=5.0, omega=0.25, v=lin_vel).T

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

    sol, h, h_2 = solve_qp_cpu(belief, goal_loc)
    # sol, h, h_2 = solve_qp(belief, goal_loc)

    # clf_values.append(V)
    cbf_values.append([h, h_2])

    u_sol = jnp.array([sol.primal[0][:2]]).reshape(-1, 1)

    # u_sol = sol
    u_opt = jnp.clip(u_sol, -U_MAX.reshape(-1,1), U_MAX.reshape(-1,1))
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
            prob_leave.append((t, estimator.prob_leaving_CVaR_GEKF(alpha, delta, beta)))

        if estimator.name == "EKF":
            estimator.update(x_measured)
            prob_leave.append((t, estimator.prob_leaving_VaR_EKF(alpha, delta, beta)))

        
    NEES_list.append(estimator.NEES(x_true))
    NIS_list.append(estimator.NIS())
    x_estimated, p_estimated = estimator.get_belief()

    eta=1.0
    goal_loc = goal_x_nom[t]

    # Store for plotting
    u_traj.append(u_opt)
    x_meas.append(x_true.at[1].set(obs_fun(x_measured)[0]))
    x_est.append(x_estimated)
    kalman_gains.append(estimator.K)
    covariances.append(p_estimated)
    in_covariances.append(estimator.in_cov)
    x_nom.append(goal_loc[:2])

with open("matrices.json", "w") as f:
    json.dump([c.tolist() for c in covariances], f, indent=2)

# Convert to numpy arrays for plotting
x_traj = np.array(x_traj).squeeze()
x_meas = np.array(x_meas).squeeze()
x_est = np.array(x_est).squeeze()
u_traj = np.array(u_traj)
cbf_values = jnp.array(cbf_values) 
x_nom = np.array(x_nom).squeeze()
time = dt*np.arange(T)  # assuming x_meas.shape[0] == N

fig, axs = plt.subplots(3, 2, figsize=(14, 18))  # 3 rows, 2 columns
axs = axs.ravel()  # flatten to 1D for easy indexing

# 1. Trajectories
axs[0].scatter(x_meas[:, 0], x_meas[:, 1], color="green", marker="o", s=1.0, alpha=0.5, label="Measured Trajectory")
axs[0].plot(x_traj[:, 0], x_traj[:, 1], "b-", label="Trajectory (True state)")
axs[0].plot(x_est[:, 0], x_est[:, 1], color="orange", label="Estimated Trajectory")
axs[0].plot(x_nom[:, 0], x_nom[:, 1], "black", label="Nominal Trajectory")
axs[0].axhline(y=wall_y, color="red", linestyle="dashed", linewidth=1, label="Obstacle")
axs[0].axhline(y=-wall_y, color="red", linestyle="dashed", linewidth=1)
axs[0].set_xlabel("x"); axs[0].set_ylabel("y")
axs[0].set_title(f"2D Trajectory ({estimator.name})")
axs[0].legend(); axs[0].grid()

# 2. Controls + barrier functions
h_vals = cbf_values[:, 0]
h2_vals = cbf_values[:, 1]
axs[1].plot(time, h_vals, color='red', label=f"y < {wall_y}")
axs[1].plot(time, h2_vals, color='purple', label=f"y > -{wall_y}")
for i in range(m):
    axs[1].plot(time, u_traj[:, i], label=f"u_{i}")
axs[1].set_xlabel("Time step (s)"); axs[1].set_ylabel("Control value")
axs[1].set_title(f"Control Values ({estimator.name})")
axs[1].legend(); axs[1].grid()

# 3. Trace of covariance
covariance_traces = [jnp.trace(P) for P in covariances]
axs[2].plot(time, np.array(covariance_traces), color="red", linestyle="-", label="Trace of Covariance")
axs[2].set_xlabel("Time Step (s)"); axs[2].set_ylabel("Trace Value")
axs[2].set_title(f"Trace of Covariance Over Time ({estimator.name})")
axs[2].legend(); axs[2].grid()

# 4. NEES with chi-square bounds
state_dim = dynamics.state_dim
confidence = 0.95
alpha = 1 - confidence
lower = chi2.ppf(alpha/2, df=state_dim)
upper = chi2.ppf(1 - alpha/2, df=state_dim)
axs[3].axhline(lower, color="red", linestyle="--", label=f"Lower {confidence*100:.1f}% bound")
axs[3].axhline(upper, color="green", linestyle="--", label=f"Upper {confidence*100:.1f}% bound")
axs[3].axhline(state_dim, color="purple", linestyle="--", label="Expected value")
axs[3].plot(time, np.array(NEES_list), color="green", linestyle="-", label="NEES")
axs[3].set_xlabel("Time Step (s)"); axs[3].set_ylabel("NEES Value")
axs[3].set_title(f"NEES over Time ({estimator.name})")
axs[3].legend(); axs[3].grid()

# 5. Probability of leaving safe set
times = np.array([t for t, _ in prob_leave])
probs = np.array([p for _, p in prob_leave])
axs[4].plot(times/1000, probs, color="blue", linestyle="dashed", label="Probs leaving (CBF 1)")
axs[4].set_title(f"Probability of Leaving/ Staying ({estimator.name})")
axs[4].set_xlabel("Time Step (s)"); axs[4].set_ylabel("Probability")
axs[4].legend(); axs[4].grid()

# 6. Leave last subplot empty or add another metric
axs[5].axis("off")  # blank

# plt.tight_layout()
fig.savefig("Sin Gain Scheduling.png", dpi=300) 

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

print("- Controls - \n")

print("Number of estimate exceedances: ", np.sum(x_est[:, 1] > wall_y))
print("Number of true exceedences", np.sum(x_traj[:, 1] > wall_y))
print("Max true value: ", np.max(x_traj[:, 1]))
print("Min true value: ", np.min(x_traj[:, 1]))
print("Max estimate value: ", np.max(x_est[:, 1]))
print("Min estimate value: ", np.min(x_est[:, 1]))

pct_h1 = 100 * np.mean(h_vals < 0)
pct_h2 = 100 * np.mean(h2_vals < 0)
print(f"Percentage h_vals < 0:  {pct_h1:.2f}%")
print(f"Percentage h2_vals < 0: {pct_h2:.2f}%")

print("- State Estimation - \n")
print(f"{estimator.name} Tracking RMSE: ", np.sqrt(np.mean((x_traj - x_est) ** 2)))
nees_arr = np.array(NEES_list)
coverage = 100 * np.mean((nees_arr >= lower) & (nees_arr <= upper))
print(f"Percentage of NEES values within {100*(1-alpha):.1f}% bounds: {coverage:.2f}%")
# --- covariance traces ---
cov_traces = np.array(covariance_traces)
mean_trace = np.mean(cov_traces)
final_trace = cov_traces[-1]
print(f"Mean trace(P):  {mean_trace:.4f}")
print(f"Final trace(P): {final_trace:.4f}")
# --- probability bounds ---
mean_prob = np.mean(probs)
max_prob  = np.max(probs)
print(f"Mean probability bound: {mean_prob:.4f}")
print(f"Max probability bound:  {max_prob:.4f}")

# Plot distance from safety boudary of estimates, max estimate value, 