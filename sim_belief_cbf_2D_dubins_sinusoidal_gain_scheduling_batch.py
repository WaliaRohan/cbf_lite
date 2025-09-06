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

# For logging results
import os
from datetime import datetime
import json

def getSimParams():

    lin_vel = 5.0
        
    sim_params = {
        "dt": 0.001,
        "T": 300,  # EKF BECOMES UNSTABLE > 30000 !!!
        "dynamics": DubinsMultCtrlDynamics(),  # or UnicycleDynamics()
        "wall_y": 5.0,
        "lin_vel": lin_vel,
        "x_init": [0.0, 0.0, lin_vel, 0.8],
        "estimator_type": "GEKF",
        "CBF_ON": True,
    }

    sensor_params = {
        "mu_u": 0.1,
        "sigma_u": jnp.sqrt(0.01),   # std deviation
        "mu_v": 0.001,
        "sigma_v": jnp.sqrt(0.0005), # std deviation
        "sensor_update_frequency": 0.1, # Hz
        "h": lambda x: jnp.array([x[1]])  # Observation Function
    }

    control_params = {
        "U_MAX": np.array([lin_vel, 0.5]),
        "clf_gain": 20.0,            # CLF linear gain
        "clf_slack_penalty": 50.0,
        "cbf_gain": 50.0,            # CBF linear gain
        "CBF_ON": True,
    }

    belief_cbf_params = {
        "alpha1": jnp.array([0.0, -1.0, 0.0, 0.0]),
        "alpha2": jnp.array([0.0,  1.0, 0.0, 0.0]),
        "beta": jnp.array([-sim_params["wall_y"]]),
        "delta": 0.001
    }

    return sim_params, sensor_params, control_params, belief_cbf_params

def simulate(sim_params, sensor_params, control_params, belief_cbf_params, key=None):
    
    # --- Unpack sim_params ---
    dt = sim_params["dt"]
    T = sim_params["T"]
    dynamics = sim_params["dynamics"]
    wall_y = sim_params["wall_y"]
    lin_vel = sim_params["lin_vel"]
    x_init = sim_params["x_init"]
    estimator_type = sim_params["estimator_type"]

    # --- Unpack sensor_params ---
    mu_u = sensor_params["mu_u"]
    sigma_u = sensor_params["sigma_u"]
    mu_v = sensor_params["mu_v"]
    sigma_v = sensor_params["sigma_v"]
    sensor_update_frequency = sensor_params["sensor_update_frequency"]
    h = sensor_params["h"]

    # --- Unpack control_params ---
    U_MAX = control_params["U_MAX"]
    clf_gain = control_params["clf_gain"]
    clf_slack_penalty = control_params["clf_slack_penalty"]
    cbf_gain = control_params["cbf_gain"]
    CBF_ON = control_params["CBF_ON"]

    # --- Unpack belief_cbf_params ---
    alpha1 = belief_cbf_params["alpha1"]
    alpha2 = belief_cbf_params["alpha2"]
    beta   = belief_cbf_params["beta"]
    delta  = belief_cbf_params["delta"]

    cbf = BeliefCBF(alpha1, beta, delta, dynamics.state_dim)
    cbf2 = BeliefCBF(alpha2, beta, delta, dynamics.state_dim)

    # Initial state (truth)
    x_true = jnp.array(x_init)  # Start position

    if sensor.__name__ == "noisy_sensor_mult":
         x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v, key=key) # mult_noise
    elif sensor.__name__ == "unbiased_noisy_sensor":
        x_initial_measurement = sensor(x_true, t=0, std=sigma_v, key=key) # unbiased_fixed_noise

    if estimator_type == "EKF":
        estimator = EKF(dynamics, dt, x_init=x_initial_measurement, R=jnp.square(sigma_v)*jnp.eye(dynamics.state_dim))
    elif estimator_type == "GEKF":
        estimator = GEKF(dynamics, dt, mu_u, sigma_u, mu_v, sigma_v, x_init=x_initial_measurement)

    # OSQP solver instance
    solver = OSQP()

    m = len(U_MAX) # control dim
    var_dim = m + 1 # ctrl dim + slack variable

    # @jit
    def solve_qp(b, goal_loc):

        x_estimated, _ = cbf.extract_mu_sigma(b)

        x_current = jnp.concatenate([x_estimated[:2], x_estimated[3:]]) # Deleting velocity fro mstate vector to match with goal_loc
        u_nom = gain_schedule_ctrl(v_r=lin_vel,
                                x = x_current ,
                                ell=0.163,
                                x_d = goal_loc,
                                lambda1=1.0, a1=16.0, a2=100.0)

        # Compute CBF components
        h = cbf.h_b(b)
        L_f_hb, _, L_f_2_h, Lg_Lf_h, _, _ = cbf.h_dot_b(b, dynamics) # ∇h(x)

        L_f_h = L_f_hb

        rhs, L_f_h, _ = cbf.h_b_r2_RHS(h, L_f_h, L_f_2_h, cbf_gain)

        # Compute CBF2 components
        h_2 = cbf2.h_b(b)
        L_f_hb_2, _, L_f_2_h_2, Lg_Lf_h_2, _, _ = cbf2.h_dot_b(b, dynamics) # ∇h(x)

        L_f_h_2 = L_f_hb_2

        rhs2, _, _ = cbf2.h_b_r2_RHS(h_2, L_f_h_2, L_f_2_h_2, cbf_gain)

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

    solve_qp_cpu = jit(solve_qp, backend='cpu')

    x_traj = []  # Store trajectory
    x_meas = [] # Measurements
    x_est = [] # Estimates
    u_traj = []  # Store controls
    cbf_values = []
    kalman_gains = []
    covariances = []
    in_covariances = [] # Innovation Covariance of EKF
    prob_leave = [] # Probability of leaving safe set
    x_nom = [] # Store nominal trajectory

    x_estimated, p_estimated = estimator.get_belief()
    x_measured = x_initial_measurement

    # Generate goal trajectory
    t_vec = jnp.arange(0.0, T + 1.0, 1.0)*dt
    goal_x_nom = s_trajectory(t_vec, A=5.0, omega=0.25, v=lin_vel).T

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
            if sensor.__name__ == "noisy_sensor_mult":
                x_measured = sensor(x_true, t, mu_u, sigma_u, mu_v, sigma_v, key=key)        
            elif sensor.__name__ == "unbiased_noisy_sensor":
                x_measured = sensor(x_true, t, sigma_v, key=key) # unbiased_fixed_noise

            if estimator.name == "GEKF":
                estimator.update(x_measured)
                prob_leave.append((t, estimator.prob_leaving_CVaR_GEKF(alpha1, delta, beta)))

            if estimator.name == "EKF":
                estimator.update(x_measured)
                prob_leave.append((t, estimator.prob_leaving_VaR_EKF(alpha1, delta, beta)))

        x_estimated, p_estimated = estimator.get_belief()

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
    cbf_values = jnp.array(cbf_values) 

    x_nom = np.array(x_nom).squeeze()

    # Define Metrics
    metrics = {
        "num_est_exceed":  np.sum(np.abs(x_est[:, 1]) > wall_y),
        "num_true_exceed": np.sum(np.abs(x_traj[:, 1]) > wall_y),
        "max_est":             np.max(x_est[:, 1]),
        "max_true":            np.max(x_traj[:, 1]),
        "tracking_rmse":       np.sqrt(np.mean((x_traj - x_est) ** 2)),
        # Add NEEMS
        # Add "Final Covariance Value"
        # Add max prob of leaving CBF 1
        # Add max prob of leaving CBF 2
        # Add max prob of staying in CBF 1
        # Add max prob of staying in CBF 2
        # Add mean prob of leaving CBF 1
        # Add mean prob of leaving CBF 2
        # Add mean prob of staying in CBF 1
        # Add mean prob of staying in CBF 2
    }

    # Plot trajectory with y-values set to zero
    plt.figure(figsize=(10, 10))
    plt.plot(x_meas[:, 0], x_meas[:, 1], color="Green", linestyle=":", label="Measured Trajectory", alpha=0.5)
    plt.plot(x_traj[:, 0], x_traj[:, 1], "b-", label="Trajectory (True state)")
    plt.plot(x_est[:, 0], x_est[:, 1], "Orange", label="Estimated Trajectory")
    plt.plot(x_nom[:, 0], x_nom[:, 1], "Green", label="Nominal Trajectory")
    plt.axhline(y=wall_y, color="red", linestyle="dashed", linewidth=1, label="Obstacle")
    plt.axhline(y=-wall_y, color="red", linestyle="dashed", linewidth=1)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("y", fontsize=14)
    plt.title(f"2D Trajectory ({estimator.name})", fontsize=14)
    plt.legend()
    plt.grid()

    # # Plot controls
    # h_vals   = cbf_values[:, 0]
    # h2_vals  = cbf_values[:, 1]
    # plt.figure(figsize=(10, 10))
    # plt.plot(time, h_vals, color='red', label=f"y < {wall_y}")
    # plt.plot(time, h2_vals, color='purple', label=f"y > -{wall_y}")
    # for i in range(m):
    #     plt.plot(time, u_traj[:, i], label=f"u_{i}")
    # plt.xlabel("Time step (s)")
    # plt.ylabel("Control value")
    # plt.title(f"Control Values ({estimator.name})")
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.legend(fontsize=14)

    # kalman_gain_traces = [jnp.trace(K) for K in kalman_gains]
    # covariance_traces = [jnp.trace(P) for P in covariances]
    # inn_cov_traces = [jnp.trace(cov) for cov in in_covariances]

    # # Plot trace of Kalman gains and covariances
    # plt.figure(figsize=(10, 10))
    # plt.plot(time, np.array(covariance_traces), color="red", linestyle="-", label="Trace of Covariance")
    # plt.xlabel("Time Step (s)")
    # plt.ylabel("Trace Value")
    # plt.title(f"Trace of Covariance Over Time ({estimator.name})")
    # plt.legend()
    # plt.grid()

    # # Probability of leaving safe set (CBF 1)
    # times  = np.array([t for t, _ in prob_leave])
    # probs  = np.array([p for _, p in prob_leave])
    # print(times)
    # dist = wall_y - x_est
    # plt.figure(figsize=(10, 10))
    # plt.plot(times/1000, probs, color="blue", linestyle="dashed", label="Probs leaving (CBF 1)")
    # plt.title(f"Probability of leaving/staying({estimator.name})")
    # plt.xlabel("Time Step (s)")
    # plt.ylabel("Distance")
    # plt.legend()
    # plt.grid()
    # plt.show()

    return metrics

def printMetrics(metrics): 
    print("\n--- Results ---")
    print("Number of estimate exceedances:",     metrics["num_est_exceed"])
    print("Number of true exceedences:",         metrics["num_true_exceed"])
    print("Max estimate value:",                 metrics["max_est"])
    print("Max true value:",                     metrics["max_true"])
    print("Tracking RMSE:",                      metrics["tracking_rmse"])

def printSimParams(sim_params, sensor_params, control_params, belief_cbf_params):
    def print_dict(d, name="dict"):
        print(f"\n{name}:")
        for key, value in d.items():
            print(f"  {key}: {value}")

    print("\n--- Simulation Parameters ---")
    print_dict(sim_params)

    print("\n-- Sensor Params ---")
    print_dict(sensor_params)
    
    print("\n--- Environment Setup ---")
    print_dict(control_params)
 
    print("\n--- Control Parameters ---")
    print_dict(belief_cbf_params)

    if sim_params["CBF_ON"]:
        print("CBF: ON")
    else:
        print("CBF: OFF")

    print("\n--- Belief CBF Parameters ---")

start_idx = 1
end_idx = 3

save_freq = 5
base_path = "/home/speedracer1702/Projects/automata_lab/cbf_lite/Results/Summer 2025/sim_belief_dubins_sinusoidal/Batch"
os.makedirs(base_path, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_path = os.path.join(base_path, f"run_EKF_nominal_{timestamp}.json")

if __name__ == "__main__":

    sim_params, sensor_params, control_params, belief_cbf_params = getSimParams()

    printSimParams(sim_params, sensor_params, control_params, belief_cbf_params)

    indices = list(range(start_idx, end_idx + 1))

    metrics_sum = {}

    for i in tqdm(indices, desc="Running simulations"):

        key = jax.random.PRNGKey(i)

        metrics = simulate(sim_params, sensor_params, control_params, belief_cbf_params, key=key)
        printMetrics(metrics) # For a single simuation

        if not metrics_sum:
            metrics_sum = {k: float(v) for k, v in metrics.items()}
        else:
            for k in metrics:
                metrics_sum[k] += float(metrics[k])

        # Save every nth run
        if (i - start_idx + 1) % save_freq == 0:
            avg_metrics = {k: v / (i - start_idx + 1) for k, v in metrics_sum.items()}
            with open(save_path, "w") as f:
                json.dump({
                    "start_idx": start_idx,
                    "end_idx": i,
                    "sim_params": {k: str(v) for k, v in sim_params.items()},
                    "sensor_params": {k: str(v) for k, v in sensor_params.items()},
                    "control_params": {k: str(v) for k, v in control_params.items()},
                    "belief_cbf_params": {k: str(v) for k, v in belief_cbf_params.items()},
                    "avg_metrics": avg_metrics
                }, f, indent=2)

    # Final average
    avg_metrics = {k: v / len(indices) for k, v in metrics_sum.items()}

    print(f"\n --- Results for runs {start_idx} to {end_idx} ({end_idx - start_idx + 1} runs) ---")
    printMetrics(avg_metrics)